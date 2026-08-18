import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


ALIGNED_PATH = "data/processed/aligned_embeddings.pt"
FUSED_PATH = "data/processed/fused_embeddings.pt"
CSV_PATH = "data/processed/multimodal_reviews.csv"


def labels_from_indices(original_indices, df):
    ratings = df.iloc[original_indices]["rating"].values
    labels = []
    for rating in ratings:
        if rating <= 2:
            labels.append("negative")
        elif rating == 3:
            labels.append("neutral")
        else:
            labels.append("positive")
    return labels


def run_experiment(name, X, y):
    """Train/evaluate with the exact same split + classifier config used
    across bert_baseline.py, clip_baseline.py, and multimodal_train.py,
    so results are directly comparable."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )

    return {
        "Experiment": name,
        "Accuracy": accuracy,
        "Precision (weighted)": precision_w,
        "Recall (weighted)": recall_w,
        "F1 (weighted)": f1_w,
        "F1 (macro)": f1_macro,
    }


print("Loading aligned embeddings (BERT + CLIP)...")
aligned_data = torch.load(ALIGNED_PATH, weights_only=False)
bert_embeddings = aligned_data["bert_embeddings"]
clip_embeddings = aligned_data["clip_embeddings"]
original_indices_aligned = aligned_data["original_indices"]

print("Loading fused embeddings (BERT + CLIP concatenated)...")
fused_data = torch.load(FUSED_PATH, weights_only=False)
fused_embeddings = fused_data["fused_embeddings"]
original_indices_fused = fused_data["original_indices"]

# Sanity check: fused and aligned files must refer to the same 995 rows
# in the same order, otherwise the three experiments aren't comparable.
assert original_indices_aligned == original_indices_fused, (
    "original_indices mismatch between aligned_embeddings.pt and "
    "fused_embeddings.pt -- the three experiments would not be using "
    "the same samples in the same order."
)

df = pd.read_csv(CSV_PATH)
y = labels_from_indices(original_indices_aligned, df)

results = []
results.append(run_experiment("Text only (BERT)", bert_embeddings.numpy(), y))
results.append(run_experiment("Image only (CLIP)", clip_embeddings.numpy(), y))
results.append(run_experiment("Text + Image (BERT+CLIP fusion)", fused_embeddings.numpy(), y))

summary_df = pd.DataFrame(results).set_index("Experiment")

print("\n===== Sentiment Ablation Summary (995 aligned samples, same split) =====\n")
print(summary_df.round(4).to_string())

OUTPUT_PATH = "data/processed/ablation_summary.csv"
summary_df.round(4).to_csv(OUTPUT_PATH)
print(f"\nSaved summary table to {OUTPUT_PATH}")
