import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error


def wmape(y_true, y_pred):
    """Weighted Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def get_all_k_metrics(total_pred):
    """
    Compute MAE and wMAPE for top k samples.

    Expects `total_pred` DataFrame to have:
    - An "Actual_Quantity" column for actual values.
    - One or more "Predicted_<Model_Name>" columns for predictions.
    """
    if "Actual_Quantity" not in total_pred.columns:
        raise ValueError("Missing 'Actual_Quantity' column in total_pred DataFrame")

    # Identify prediction columns (ignore the actual values)
    model_names = [col for col in total_pred.columns if col.startswith("Predicted_")]
    
    # Define metric labels
    index = ['MAE@5', 'MAE@10', 'MAE@20', 'wMAPE@5', 'wMAPE@10', 'wMAPE@20']
    final_res = pd.DataFrame(columns=model_names, index=index)
    
    for model in model_names:
        for k in [5, 10, 20]:
            # Select top-k actual values
            pred_at_k = total_pred.nlargest(k, 'Actual_Quantity')
            
            # Compute MAE and wMAPE
            mae_k = mean_absolute_error(pred_at_k['Actual_Quantity'], pred_at_k[model])
            wmape_k = wmape(pred_at_k['Actual_Quantity'], pred_at_k[model])

            # Store results
            final_res.loc[f"MAE@{k}", model] = round(mae_k, 2)
            final_res.loc[f"wMAPE@{k}", model] = round(wmape_k, 3)
    
    return final_res
