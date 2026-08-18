import torch

FILE_PATH = "data/processed/image_embeddings.pt"

data = torch.load(FILE_PATH, weights_only=False)

embeddings = data["embeddings"]
image_paths = data["image_paths"]

print("Keys:", data.keys())
print("Embeddings shape:", embeddings.shape)
print("No.of image paths:", len(image_paths))

print("/nFirst 3 embeddings:")
for path in image_paths[:3]:
    print(path)

print("\nFirst image embedding:")
print(embeddings[0])

print("\nFirst embedding shape:", embeddings[0].shape)