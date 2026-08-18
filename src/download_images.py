import os
import pandas as pd
import requests

CSV_FILE_PATH = "data/processed/multimodal_reviews.csv"
IMAGES_DIR = "data/images"

os.makedirs(IMAGES_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE_PATH)

print(f"Found {len(df)} reviews with images. Starting download...")

downloaded_count = 0

for index, row in df.iterrows():
    image_url = row["image_url"]
    image_path = os.path.join(IMAGES_DIR, f"{row['asin']}_{index}.jpg")

    try:
        response = requests.get(image_url, timeout=10)

        if response.status_code == 200:
            with open(image_path, "wb") as f:
                f.write(response.content)

            print(f"Downloaded: {image_path} ")
            downloaded_count += 1

        else:
            print(f"Failed ({response.status_code}) : {image_url}")

    except Exception as e:
        print(f"Error downloading image {index}: {e}")

print(f"\nFinished downloading {downloaded_count} images")