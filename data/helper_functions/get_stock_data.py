import yfinance as yf

# Get SP500 index (symbol: ^GSPC)
sp500 = yf.download("^GSPC", start="1980-01-01", end="2018-04-20")

print(sp500.head())
sp500.to_csv("sp500_1980_present.csv")

import pandas as pd

# Define the actual columns you care about
columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

# Skip the first 3 rows and set your own column names
df = pd.read_csv("sp500_1980_present.csv", skiprows=3, names=columns)

# Keep only Date, Close, and Volume
df = df[["Date", "Close", "Volume"]]

# Save cleaned version
df.to_csv("sp500_clean.csv", index=False)

print(df.head())
print("✅ Saved to sp500_clean.csv")
