import pandas as pd
from src.run import hparams
from src.config import LOG_DIR

EMB_DIM = 100

RESULTS_PREFIX = hparams['results_prefix']
EMBEDDINGS_TRAIN_GAN_PATH = LOG_DIR + "/"+ RESULTS_PREFIX + "_train_gan_embeddings.pkl"
EMBEDDINGS_TEST_GAN_PATH = LOG_DIR + "/" + RESULTS_PREFIX + "_test_gan_embeddings.pkl"

def create_embeddings_df(embeddings_df, train_df, test_df, categ_data, limit_size=1):
    # Reset index to bring 'date' back as a column
    categ_data = categ_data.reset_index()
    # Ensure dates are datetime (do not convert to string)
    categ_data['date'] = pd.to_datetime(categ_data['date'])
    embeddings_df['date'] = pd.to_datetime(embeddings_df['date'])
    train_df = train_df.reset_index()
    test_df = test_df.reset_index()
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])

    # Merge embeddings with categ_data on datetime values
    gen_data = categ_data.merge(embeddings_df, on='date', how='left')
    gen_data = gen_data.fillna(method='ffill', limit=limit_size)  # Fill missing values
    gen_data = gen_data.fillna(-1)

    # If the embeddings column exists, expand it into emb_0,...,emb_99
    if 'embeddings' in gen_data.columns:
        gen_data['embeddings'] = gen_data['embeddings'].apply(lambda x: [0.0] * EMB_DIM if x == -1 else x)
        for ii in range(EMB_DIM):
            gen_data[f'emb_{ii}'] = gen_data['embeddings'].apply(lambda x: x[ii])

    # Drop unnecessary columns
    drop_cols = [col for col in ['Quantity', 'embeddings'] if col in gen_data.columns]
    X_gen = gen_data.drop(columns=drop_cols)

    # Set 'date' as index
    if 'date' in X_gen.columns:
        X_gen = X_gen.set_index('date')

    # Ensure train_df and test_df have datetime indexes
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    train_df.set_index('date', inplace=True)
    test_df.set_index('date', inplace=True)

    # Merge train & test data with embeddings (using datetime as the key)
    train_set = train_df.merge(X_gen, left_index=True, right_index=True, how='left')
    test_set = test_df.merge(X_gen, left_index=True, right_index=True, how='left')

    return train_set.reset_index(), test_set.reset_index()



def get_gan_embeddings(train_df, test_df, categ_data, window_size=2):
    train_gan_df = pd.read_pickle(EMBEDDINGS_TRAIN_GAN_PATH)
    test_gan_df = pd.read_pickle(EMBEDDINGS_TEST_GAN_PATH)
    print("PRINTING EMBEDDINGS PATHS:/n")
    print(EMBEDDINGS_TRAIN_GAN_PATH)
    print(EMBEDDINGS_TEST_GAN_PATH)
    embeddings_df = pd.concat([train_gan_df, test_gan_df], ignore_index=True)
    train_set, test_set = create_embeddings_df(embeddings_df, train_df, test_df, categ_data, limit_size=(window_size-1))
    return train_set, test_set