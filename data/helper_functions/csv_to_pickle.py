import pandas as pd
import pickle
import ast
import numpy as np
import re

def fix_embedding(embedding_str):
    """
    Fixes malformed embedding strings by inserting missing commas
    and converting it into a proper list of floats.
    """
    if isinstance(embedding_str, str):
        # Ensure proper list format: add missing commas between numbers
        fixed_str = re.sub(r'(?<=\d)\s+(?=-?\d)', ', ', embedding_str.strip())  
        return np.array(ast.literal_eval(fixed_str), dtype=np.float32)
    return embedding_str  # If already converted, return as is

def csv_to_pkl(csv_filename, pkl_filename):
    """
    Converts a CSV file into a .pkl file, ensuring that the 'embedding' column 
    is stored as a numerical vector
    """
    try:
        df = pd.read_csv(csv_filename)

        # Convert the 'embedding' column from string to list of floats
        if 'embedding' in df.columns:
            df['embedding'] = df['embedding'].apply(fix_embedding)

        assert isinstance(df['embedding'].iloc[0], (np.ndarray, list)), \
            f"Embedding not parsed correctly! Got {type(df['embedding'].iloc[0])}"

        with open(pkl_filename, 'wb') as pkl_file:
            pickle.dump(df, pkl_file)

        print(f"Successfully saved {csv_filename} as {pkl_filename} with proper embedding format.")
    
    except Exception as e:
        print(f"Error: {e}")

csv_filename = "../world_events_dataset_from_1980_3.0.csv"  
pkl_filename = "../world_events_dataset_from_1980_3.0.pkl"

df = pd.read_csv(csv_filename)
print("Before Fixing Embeddings:")
print(df.head())

csv_to_pkl(csv_filename, pkl_filename)
# check if it worked
with open(pkl_filename, 'rb') as f:
    df_fixed = pickle.load(f)

print("\nAfter Fixing Embeddings:")
print(df_fixed.head())
print("Type of first embedding:", type(df_fixed['embedding'].iloc[0]))