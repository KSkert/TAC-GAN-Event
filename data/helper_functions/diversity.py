import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import re
from scipy.stats import entropy
from collections import Counter

file_path = "../Base_events_5.0.csv"
df = pd.read_csv(file_path)

# parse the 'embedding' column 
def parse_embedding(embedding_str):
    number_pattern = r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+"
    values = re.findall(number_pattern, embedding_str)
    if len(values) != 100:
        raise ValueError(f"Unexpected embedding length: found {len(values)} values")
    return np.array([float(v) for v in values])

parsed_embeddings = []
bad_rows = []

for idx, raw_str in enumerate(df["embedding"]):
    try:
        parsed_embeddings.append(parse_embedding(raw_str))
    except Exception as e:
        bad_rows.append((idx, str(e)))

if bad_rows:
    print(f"Skipped {len(bad_rows)} malformed embeddings:")
    for idx, err in bad_rows[:5]:
        print(f" - Row {idx}: {err}")

embeddings = np.vstack(parsed_embeddings)

# pairwise calculations
cos_sim_matrix = cosine_similarity(embeddings)
eucl_dist_matrix = euclidean_distances(embeddings)

n = cos_sim_matrix.shape[0]
mask = ~np.eye(n, dtype=bool)

cos_sim_values = cos_sim_matrix[mask]
eucl_dist_values = eucl_dist_matrix[mask]

# Gini coefficient
def gini(x):
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))

gini_cosine = gini(cos_sim_values)
print(f"Gini Coefficient (Cosine Similarity): {gini_cosine:.4f}")

# 95th percentile of eucl distances (outliers)
eucl_95 = np.percentile(eucl_dist_values, 95) 
print(f"95th Percentile Euclidean Distance: {eucl_95:.4f}")

# Entropy for label columns
def compute_entropy(column_name):
    labels = df[column_name].astype(str)
    counts = Counter(labels)
    probs = np.array(list(counts.values())) / len(labels)
    raw_entropy = entropy(probs, base=2)

    unique_labels = len(counts)
    max_entropy = np.log2(unique_labels) if unique_labels > 1 else 1
    normalized_entropy = raw_entropy / max_entropy if max_entropy else 0

    print(f"'{column_name}' Label Entropy: {raw_entropy:.4f} bits")
    print(f"'{column_name}' Normalized Entropy: {normalized_entropy:.4f}")

compute_entropy("High-Category")
compute_entropy("Category")