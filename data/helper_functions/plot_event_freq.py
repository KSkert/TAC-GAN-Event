import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the CSV and parse dates
input_file = "../Base_events_3.0.csv"
df = pd.read_csv(input_file, parse_dates=["date"])

# 2. Define the date range for filtering
start_date = pd.Timestamp("1980-01-01")
end_date = pd.Timestamp("2018-04-20")
mask = (df["date"] >= start_date) & (df["date"] <= end_date)
filtered_df = df[mask]

# 3. Count the number of events per day
daily_counts = filtered_df.groupby("date").size().reset_index(name="count")

# 4. Plot the frequency of daily events
plt.figure(figsize=(12, 6))
plt.plot(daily_counts["date"], daily_counts["count"])
plt.title("Daily Events from 1980-01-01 to 2018-04-20")
plt.xlabel("Date")
plt.ylabel("Number of Events")
plt.tight_layout()
plt.show()

# 5. Print top 10 days with the highest event counts
top_days = daily_counts.sort_values(by="count", ascending=False).head(10)
print("Top 10 days with the most events:")
print(top_days.to_string(index=False))

# 6. Calculate average and standard deviation
mean_count = daily_counts["count"].mean()
std_count = daily_counts["count"].std()

print(f"\n📊 Average number of events per day: {mean_count:.2f}")
print(f"📈 Standard deviation of events per day: {std_count:.2f}")

# 7. Print how many rows fell in the date range
row_count = len(filtered_df)
print(f"\nTotal number of rows between {start_date.date()} and {end_date.date()}: {row_count}")
