import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

CSV_FILE_PATH = "data/processed/multimodal_reviews.csv"
OUTPUT_PATH = "data/processed/bert_embeddings.pt"

MODEL_NAME = "bert-base-uncased"

print("Loading multimodal review dataset...")

df = pd.read_csv(CSV_FILE_PATH)

#Remove reviews without text
df = df.dropna(subset=["text"])

print(f"Found {len(df)} reviews with text.")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME).to(device)

model.eval()

embeddings = []
texts = []

print("Generating BERT embeddings for review texts...")

for text in tqdm(df["text"], desc="Generating BERT embeddings"):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

        with torch.no_grad():
            output = model(**inputs)

        embedding = output.last_hidden_state[:, 0, :].cpu()  # Use the [CLS] token representation

        embeddings.append(embedding)
        texts.append(text)
    except Exception as e:
        print(f"Error processing text: {text}. Error: {e}")

if embeddings:
    embeddings_tensor = torch.cat(embeddings, dim=0)
    torch.save({"embeddings": embeddings_tensor, "texts": texts}, OUTPUT_PATH)
    print(f"Saved {len(embeddings)} BERT embeddings to {OUTPUT_PATH}")

    print("\nfinished!")
    print(f"Reviews processed: {len(embeddings)}")
    print(f"Embeddings shape: {embeddings_tensor.shape}")
    print(f"Embeddings saved to: {OUTPUT_PATH}")
else:
    print("No embeddings were generated. Please check the review dataset and ensure it contains valid text entries.")