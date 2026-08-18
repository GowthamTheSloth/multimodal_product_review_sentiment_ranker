import torch

ALIGNED_PATH = "data/processed/aligned_embeddings.pt"
OUTPUT_PATH = "data/processed/fused_embeddings.pt"

print("Loading aligned embeddings...")

data = torch.load(ALIGNED_PATH, weights_only=False)

bert_embeddings = data["bert_embeddings"]
clip_embeddings = data["clip_embeddings"]
image_paths = data["image_paths"]
original_indices = data["original_indices"]

print(f"Aligned BERT embeddings shape: {bert_embeddings.shape}")
print(f"Aligned CLIP embeddings shape: {clip_embeddings.shape}")

#Concatenate BERT and CLIP embeddings along the feature dimension
fused_embeddings = torch.cat((bert_embeddings, clip_embeddings), dim=1)

print(f"Fused embeddings shape: {fused_embeddings.shape}")

#Save fused embeddings
torch.save({"fused_embeddings": fused_embeddings, "image_paths": image_paths, "original_indices": original_indices}, OUTPUT_PATH)

print(f"\nSaved fused embeddings to {OUTPUT_PATH}")