import json
import pandas as pd

FILE_PATH = "data/raw/All_Beauty.jsonl"
OUTPUT_PATH = "data/processed/multimodal_reviews.csv"

MAX_REVIEWS = 1000  # Limit the number of reviews to process

reviews = []
reviews_checked = 0

print("Scanning reviews for images...")

with open(FILE_PATH, "r", encoding="utf-8") as file:
    for line in file:
        review = json.loads(line)
        reviews_checked += 1

        if not review.get("images"):
            continue  # Skip reviews without images

        image = review["images"][0]

        reviews.append({
            "rating": review["rating"],
            "title": review["title"],
            "text": review["text"],
            "image_url": image["medium_image_url"],
            "asin": review["asin"],
            "helpful_vote": review["helpful_vote"],
            "verified_purchase": review["verified_purchase"],
            })

        if len(reviews) >= MAX_REVIEWS:
            break  # Stop after reaching the maximum number of reviews


df = pd.DataFrame(reviews)
df.to_csv(OUTPUT_PATH, index=False)

print("\nFinished!")
print(f"Processed {len(reviews)} reviews with images out of {reviews_checked} checked.")
print(f"Saved to {OUTPUT_PATH}")

print("\nFirst row:")
print(df.iloc[0])