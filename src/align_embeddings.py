import os
import re
import torch

BERT_PATH = "data/processed/bert_embeddings.pt"
CLIP_PATH = "data/processed/image_embeddings.pt"
OUTPUT_PATH = "data/processed/aligned_embeddings.pt"

print("Loading embeddings...")

bert_data = torch.load(BERT_PATH, weights_only=False)
clip_data = torch.load(CLIP_PATH, weights_only=False)

bert_embeddings = bert_data["embeddings"]
clip_embeddings = clip_data["embeddings"]
image_paths = clip_data["image_paths"]

print(f"BERT embeddings shape: {bert_embeddings.shape}")
print(f"CLIP embeddings shape: {clip_embeddings.shape}")

#Extract original CSV row index from each image filename
# Example: B00R8DXL44_0.jpg -> 0
image_indices = []

for image_path in image_paths:
    filename = os.path.basename(image_path)

    match = re.search(r'_(\d+)\.(jpg|jpeg|png)$', filename, re.IGNORECASE)
    if match:
        index = int(match.group(1))
        image_indices.append(index)
    else:
        print(f"Warning: Could not extract index from {filename}")

#Sort images by original CSV row index to align with BERT embeddings
paired_data = sorted(zip(image_indices, clip_embeddings, image_paths), key=lambda x: x[0])

image_indices = [item[0] for item in paired_data]
aligned_clip_embeddings = torch.stack([item[1] for item in paired_data])
aligned_image_paths = [item[2] for item in paired_data]

#Select the corresponding BERT embeddings
aligned_bert_embeddings = bert_embeddings[image_indices]

print("\nAlignment completed")

print(f"Aligned BERT embeddings shape: {aligned_bert_embeddings.shape}")
print(f"Aligned CLIP embeddings shape: {aligned_clip_embeddings.shape}")
print(f"Matched Samples: {len(image_indices)}")

#Save aligned embeddings
torch.save({
    "bert_embeddings": aligned_bert_embeddings,
    "clip_embeddings": aligned_clip_embeddings,
    "image_paths": aligned_image_paths,
    "original_indices": image_indices
}, OUTPUT_PATH)

print(f"\nSaved aligned embeddings to {OUTPUT_PATH}")

print(f"\nFirst 5 alignments:")

for i in range(min(5, len(image_indices))):
    print(f"{i}: " f"CSV row {image_indices[i]} -> " f"{os.path.basename(aligned_image_paths[i])} -> " f"BERT embedding shape: {aligned_bert_embeddings[i].shape}, " f"CLIP embedding shape: {aligned_clip_embeddings[i].shape}")
