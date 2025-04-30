import pandas as pd
import numpy as np
from collections import defaultdict


df = pd.read_csv("Base_events_4.0_shifted.csv", parse_dates=["date"])

# Parameters
TARGET_MEAN = 2.25
TARGET_STD = 3.42
TARGET_CATEGORIES = ["SocialEvent", "PoliticalEvent", "CriminalEvent", "SportsEvent"]


global_counts = defaultdict(int) # Track global counts

def balanced_sample_with_reallocation(group, n):
    if len(group) <= n or n == 0:
        return group

    # What's present on this day
    day_cats = group["High-Category"].value_counts()
    available_cats = list(day_cats.index.intersection(TARGET_CATEGORIES))

    # Categories not present
    missing_cats = list(set(TARGET_CATEGORIES) - set(available_cats))
    
    if not available_cats:
        return pd.DataFrame()

    # Base allocation
    base_allocation = {cat: n // len(TARGET_CATEGORIES) for cat in TARGET_CATEGORIES}
    remainder = n - sum(base_allocation.values())

    # Distribute remainder round-robin style
    for i in range(remainder):
        base_allocation[TARGET_CATEGORIES[i % len(TARGET_CATEGORIES)]] += 1

    # Remove missing categories and reallocate
    reallocated = 0
    for cat in missing_cats:
        reallocated += base_allocation.pop(cat, 0)

    # Reallocate intelligently to available cats to favor those with low global count (anomalies)
    if reallocated > 0:
        available_sorted = sorted(available_cats, key=lambda c: global_counts[c])
        for i in range(reallocated):
            target = available_sorted[i % len(available_sorted)]
            base_allocation[target] = base_allocation.get(target, 0) + 1

    # Sample available ones
    sampled = []
    for cat, count in base_allocation.items():
        subset = group[group["High-Category"] == cat]
        actual_count = min(len(subset), count)
        if actual_count > 0:
            sample = subset.sample(actual_count, random_state=42)
            sampled.append(sample)
            global_counts[cat] += len(sample)

    return pd.concat(sampled) if sampled else pd.DataFrame()

# Main daily allocation loop
cleaned_rows = []
for date, group in df.groupby("date"):
    sample_n = np.random.poisson(TARGET_MEAN)
    cleaned = balanced_sample_with_reallocation(group, sample_n)
    if not cleaned.empty:
        cleaned_rows.append(cleaned)

cleaned_df = pd.concat(cleaned_rows).sample(frac=1, random_state=42).reset_index(drop=True)

# Report
daily_counts = cleaned_df.groupby("date").size()
print("Cleaned average:", round(daily_counts.mean(), 2))
print("Cleaned std dev:", round(daily_counts.std(), 2))

print("\nHigh-Category distribution:")
total = sum(global_counts.values())
for cat in TARGET_CATEGORIES:
    count = global_counts[cat]
    print(f"{cat}: {count} ({count / total:.2%})")


import matplotlib.pyplot as plt
plt.hist(daily_counts, bins=20)
plt.title("Distribution of Events Per Day")
plt.xlabel("Events")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# Save
cleaned_df.to_csv("../Base_events_5.0.csv", index=False)
print("\nReallocation complete")
