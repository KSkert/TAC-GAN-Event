import pandas as pd
from wikipedia2vec import Wikipedia2Vec
from urllib.parse import unquote
import re
from numpy import mean

# Load the Wikipedia2Vec model
wiki2vec = Wikipedia2Vec.load("../../enwiki_20180420_100d.pkl")

# Load dataset
df = pd.read_csv("dbpedia2.0/all_events.csv")

# Filter to events before cutoff
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df[df["date"] < "2018-04-20"]

# Unquote and clean event labels for use as article titles
df["article_title"] = df["eventLabel"].apply(lambda x: unquote(x) if isinstance(x, str) else x)
df["article_title"] = df["article_title"].str.replace("_", " ", regex=False)

# Function: try entity or word vector
def try_get_embedding(title):
    if not isinstance(title, str):
        return None
    try:
        return wiki2vec.get_entity_vector(title).tolist()
    except KeyError:
        pass
    try:
        return wiki2vec.get_word_vector(title).tolist()
    except KeyError:
        pass
    return None

# Fallback: average word embeddings from a string
def avg_word_embedding(text):
    words = re.findall(r'\w+', text)
    vectors = []
    for word in words:
        try:
            vectors.append(wiki2vec.get_word_vector(word))
        except KeyError:
            continue
    if vectors:
        return mean(vectors, axis=0).tolist()
    return None

# Smart embedding extractor with variations
def get_embedding(row):
    candidates = []

    if isinstance(row["article_title"], str):
        candidates.append(row["article_title"])

    # Smart string variations
    for c in candidates[:]:
        candidates.append(c.title())
        candidates.append(c.replace("’", "'"))
        candidates.append(re.sub(r'\s*\(.*?\)', '', c))   # remove parentheticals
        candidates.append(re.sub(r'^\d{4}\s+', '', c))    # remove leading year
        candidates.append(re.sub(r',\s*\d{4}$', '', c))   # remove trailing year

    for candidate in candidates:
        emb = try_get_embedding(candidate)
        if emb:
            return emb

    # Fallback to average of word vectors
    return avg_word_embedding(row["article_title"])

# Apply embedding extraction
df["embedding"] = df.apply(get_embedding, axis=1)

# Split successes and failures
success_df = df[df["embedding"].notnull()].copy()
fail_df = df[df["embedding"].isnull()].copy()

# Format embedding: wrap in square brackets with space after commas
success_df["embedding"] = success_df["embedding"].apply(lambda x: f"[{', '.join(map(str, x))}]")

# Drop columns
success_df = success_df.drop(columns=["event", "eventLabel"])

# Rename columns
success_df = success_df.rename(columns={
    "countryLabel": "country",
    "article_title": "wiki_name"
})

# Save results
success_df.to_csv("dbpedia2.0/all_events_embedded.csv", index=False)
fail_df.to_csv("special_treatment.csv", index=False)

print(f"✅ Saved embeddings for {len(success_df)} events.")
print(f"⚠️ {len(fail_df)} events saved to special_treatment.csv for manual treatment.")
