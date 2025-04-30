import pandas as pd

df = pd.read_csv("wikidata_events.csv")

# Drop rows with missing/malformed dates
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

# Filter by year
df = df[(df['date'].dt.year >= 1980) & (df['date'].dt.year <= 2018)]
# Drop rows missing critical info
df = df.dropna(subset=["event", "eventLabel", "article", "typeLabel"])

# Remove dupes
df = df.drop_duplicates(subset=["event"])
df = df[df['article'].str.contains("en.wikipedia.org/wiki/")]
df.to_csv("cleaned_wikidata_events.csv", index=False)