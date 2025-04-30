import sys
import os
import torch
torch.set_default_dtype(torch.float32)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from src.demand_prediction.gan_events import get_gan_embeddings
from src.demand_prediction.events_models import save_events_model, load_events_model
from src.demand_prediction.event_lstm import EventRNNModel
from src.config import SEED

print(f"Parent class of EventRNNModel: {EventRNNModel.__bases__}")

def lstm_model_results(train,
                       test,
                       train_df,
                       test_df,
                       train_grr, 
                       test_grr,
                       categ_data,
                       start_pred_time,
                       leaf_name,
                       n_in,
                       window_size=2,
                       device='cpu',
                       lstm_df_cache=False,
                       use_cache=False,
                       use_covariates=False,
                       mode=None,
                       concat_features=None
                       ):
    """
    Trains an EventRNNModel on the 'Quantity' (target) column,
    and uses GAN-learned embeddings as future covariates to help predict unusual events.
    """

    # 1) Obtain (train_set, test_set) of day-embeddings
    print("getting embeddings...")
    train_set, test_set = get_gan_embeddings(train_df, test_df, categ_data, window_size=window_size)

    # Ensure 'date' is a column, not an index
    if train_set.index.name == 'date':
        train_set = train_set.reset_index()
    if test_set.index.name == 'date':
        test_set = test_set.reset_index()

    # Keep only date + emb_(0..99)
    emb_cols = [f"emb_{ii}" for ii in range(100)]
    train_set = train_set[['date'] + emb_cols]
    test_set  = test_set[['date'] + emb_cols]

    # 2) Scale only the target (Quantity) with a Darts Scaler
    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)  
    test_transformed = transformer.transform(test)        

    # Convert back to DataFrame and reset index to expose the date column
    train_scaled_df = train_transformed.pd_dataframe().reset_index().rename(columns={'index': 'date'})
    test_scaled_df  = test_transformed.pd_dataframe().reset_index().rename(columns={'index': 'date'})
    
    # Merge with the GAN embeddings DataFrames
    train_merged = train_scaled_df.merge(train_set, on='date', how='left')
    test_merged  = test_scaled_df.merge(test_set,  on='date', how='left')

    # Fill missing embedding days with zero
    train_merged[emb_cols] = train_merged[emb_cols].fillna(0)
    test_merged[emb_cols]  = test_merged[emb_cols].fillna(0)

    # Merge global relevance rank if provided (not used in final thesis report)
    if train_grr and test_grr:
        grr_train = pd.DataFrame({"date": pd.to_datetime(train_df.index), "global_relevance_rank": train_grr})
        grr_test = pd.DataFrame({"date": pd.to_datetime(test_df.index), "global_relevance_rank": test_grr})
        train_merged = train_merged.merge(grr_train, on="date", how="left")
        test_merged = test_merged.merge(grr_test, on="date", how="left")

    # Convert and sort date column
    train_merged['date'] = pd.to_datetime(train_merged['date'])
    test_merged['date'] = pd.to_datetime(test_merged['date'])

    # 3) Build "target" and "future covariates" TimeSeries
    train_target = TimeSeries.from_dataframe(train_merged[['date', 'Quantity']], time_col='date', value_cols='Quantity').astype(np.float32)
    train_futcov = TimeSeries.from_dataframe(train_merged[['date'] + emb_cols], time_col='date', value_cols=emb_cols).astype(np.float32)
    test_target = TimeSeries.from_dataframe(test_merged[['date', 'Quantity']], time_col='date', value_cols='Quantity').astype(np.float32)
    test_futcov = TimeSeries.from_dataframe(test_merged[['date'] + emb_cols], time_col='date', value_cols=emb_cols).astype(np.float32)

    future_covariates = train_futcov.append(test_futcov)

    # 4) Train model with EventRNNModel
    name_path_model = leaf_name + "Predicted" + start_pred_time
    lstm_model = load_events_model(name_path_model) if use_cache else None

    if lstm_model is None or not use_cache:

        # Create a fresh model using EventRNNModel
        lstm_model = EventRNNModel(
            model='LSTM',               
            n_epochs=100,
            input_chunk_length=n_in,
            output_chunk_length=1,    
            dropout=0.3,  
            optimizer_kwargs={'lr': 0.001},
            log_tensorboard=True,
            model_name=name_path_model,
            random_state=SEED
        )

        
        if use_covariates:
            lstm_model.fit(series=train_target, future_covariates=future_covariates, verbose=True)
        else:
            lstm_model.fit(series=train_target, verbose=True)

        save_events_model(lstm_model, name_path_model)

    # 5) Iterative prediction
    test_ts = train_target[-n_in:]
    prediction_time = len(test)

    if use_covariates:
        cur_forecast = lstm_model.predict(n=prediction_time, series=test_ts, future_covariates=future_covariates)
    else:
        cur_forecast = lstm_model.predict(n=prediction_time, series=test_ts)

    # Keep only the first dimension (removing additional event data)
    forecast = TimeSeries.from_times_and_values(cur_forecast.time_index, cur_forecast.values()[:, 0])

    # Invert scaling
    forecast_df = forecast.pd_dataframe().rename(columns={'0': 'Predicted_GAN - Event LSTM'})
    forecast_df.reset_index(drop=False, inplace=True)
    forecast_df.rename(columns={'index': 'date'}, inplace=True)

    forecast_timeseries = TimeSeries.from_dataframe(forecast_df, time_col='date', freq='B')
    forecast_inverted = transformer.inverse_transform(forecast_timeseries)

    # Final predictions
    predictions = forecast_inverted.pd_dataframe().rename(columns={'Quantity': 'Predicted_GAN - Event LSTM'})

    # 6) Combine with actual (unscaled) test data
    y_test_df = pd.DataFrame(test.pd_dataframe(), index=test.pd_dataframe().index, columns=['Quantity']).rename(columns={'Quantity': 'Real Quantity'})
    df_test_results = pd.concat([y_test_df, predictions], axis=1)

    df_test_results.iplot(
        title=leaf_name + " - GAN - Event LSTM",
        xTitle='Date',
        yTitle='Sales',
        theme='white'
    )

    return predictions


def get_lstm_results(train,
                     test,
                     train_df,
                     test_df,
                     train_grr, 
                     test_grr,
                     categ_data,
                     start_pred_time,
                     leaf_name,
                     n_in,
                     window_size,
                     device='cpu',
                     lstm_df_cache=False,
                     use_cache=False,
                     use_covariates=False, 
                     mode=None,
                     concat_features=None
                     ):
    """
    Simple wrapper to call lstm_model_results() with the same arguments.
    """
    return lstm_model_results(
        train,
        test,
        train_df,
        test_df,
        train_grr, 
        test_grr,
        categ_data,
        start_pred_time,
        leaf_name,
        n_in,
        window_size,
        device=device,
        lstm_df_cache=lstm_df_cache,
        use_cache=use_cache,
        use_covariates=use_covariates,
        mode=mode,
        concat_features=concat_features
    )
