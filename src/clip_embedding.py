import os
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

IMAGE_DIR = "data/images"
OUTPUT_PATH = "data/processed/image_embeddings.pt"

MODEL_NAME = "openai/clip-vit-base-patch32"

print("Loading CLIP model and processor...")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model.eval()

print(f"Using device: {device}")

embeddings = []
image_paths = []

image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

print(f"Found {len(image_files)} images. Processing...")

for image_file in tqdm(image_files, desc="Generating CLIP embeddings"):
    image_path = os.path.join(IMAGE_DIR, image_file)
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.get_image_features(**inputs)

            if hasattr(output, "pooler_output"):
                image_embedding = output.pooler_output
            else:
                image_embedding = output
        embeddings.append(image_embedding.cpu())
        image_paths.append(image_path)
    except Exception as e:
        print(f"Error processing {image_file}: {e}")

if embeddings:
    embeddings_tensor = torch.cat(embeddings, dim=0)
    torch.save({"embeddings": embeddings_tensor, "image_paths": image_paths}, OUTPUT_PATH)
    print(f"Saved {len(embeddings)} image embeddings to {OUTPUT_PATH}")

    print("\nfinished!")
    print(f"Images processed: {len(embeddings)}")
    print(f"Embeddings shape: {embeddings_tensor.shape}")
    print(f"Embeddings saved to: {OUTPUT_PATH}")
else:
    print("No embeddings were generated. Please check the image directory and ensure it contains valid image files.")