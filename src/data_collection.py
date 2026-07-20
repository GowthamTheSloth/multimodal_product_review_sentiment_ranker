import pandas as pd


df = pd.read_csv(
    "data/raw/Amazon_Reviews.csv",
    engine="python"
)

print(df["Rating"].head(10))
print(df["Rating"].unique()[:20])