import pandas as pd

df = pd.read_csv("data/raw/Amazon_Reviews.csv", engine="python")

#Extract number from ratings
df["rating_num"] = df['Rating'].str.extract(r"(\d)").astype(float)

#Convert ratings to sentiments
def rating_to_sentiment(rating):
    if pd.isna(rating):
        return None
    
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"

df["Sentiment"] = df["rating_num"].apply(rating_to_sentiment)

print(df[["Rating", "rating_num", "Sentiment"]].head(10))

#Sentiment distribution
print(df["Sentiment"].value_counts())

#Count no.of empty rows
print(df[["Rating", "rating_num", "Sentiment"]].isnull().sum())

#Remove rows with missing Sentiment
df = df.dropna(subset=["Sentiment"])

#Check Text Columns
print("Missing Review Text:", df["Review Text"].isnull().sum())
print("Missing Review Titles:", df["Review Title"].isnull().sum())

df.to_csv("data/processed/processed_reviews.csv", index=False)
print("Processed Dataset Saved")