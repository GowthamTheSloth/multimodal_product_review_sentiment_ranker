import os
import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from xgboost import XGBRanker


ALIGNED_PATH = "data/processed/aligned_embeddings.pt"
CSV_PATH = "data/processed/multimodal_reviews.csv"
MODEL_OUTPUT_PATH = "models/helpfulness_ranker.json"

# Fixed (non-data-derived) cutoffs for helpful_vote -> ordinal relevance grade.
# 0 votes -> 0, 1-2 -> 1, 3-5 -> 2, 6-10 -> 3, 11+ -> 4
RELEVANCE_BINS = [-1, 0, 2, 5, 10, 10**9]
RELEVANCE_LABELS = [0, 1, 2, 3, 4]


def helpful_vote_to_relevance(vote_series):
    return pd.cut(vote_series, bins=RELEVANCE_BINS, labels=RELEVANCE_LABELS).astype(int)


print("Loading aligned BERT embeddings (text features for ranking)...")
aligned_data = torch.load(ALIGNED_PATH, weights_only=False)
bert_embeddings = aligned_data["bert_embeddings"]
original_indices = aligned_data["original_indices"]

print(f"BERT embeddings shape: {bert_embeddings.shape}")

df = pd.read_csv(CSV_PATH)
helpful_votes = df.iloc[original_indices]["helpful_vote"].values
relevance = helpful_vote_to_relevance(pd.Series(helpful_votes)).values

print("\nRelevance grade distribution (whole aligned set):")
print(pd.Series(relevance).value_counts().sort_index())

# The first pass used BERT embeddings alone and only reached ~0.54 pairwise
# accuracy (barely above the 0.50 random baseline). Review-helpfulness
# research consistently finds that structural/metadata signals -- review
# length, how extreme the star rating is, verified purchase status --
# predict helpfulness better than semantic content alone, since helpful
# votes are largely driven by "is this review substantial and is the
# opinion strong" rather than pure writing quality. We add those as extra
# columns alongside the BERT embedding rather than replacing it.
review_text = df.iloc[original_indices]["text"].fillna("")
text_length = review_text.apply(lambda t: len(str(t).split())).values

ratings = df.iloc[original_indices]["rating"].values
rating_extremity = np.abs(ratings - 3)  # 1/5 star reviews score higher than 3 star

verified_purchase = (
    df.iloc[original_indices]["verified_purchase"]
    .astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0).values
)

meta_features = np.column_stack([text_length, ratings, rating_extremity, verified_purchase])

print(f"\nMeta feature shape: {meta_features.shape} "
      f"(columns: text_length, rating, rating_extremity, verified_purchase)")

X = np.hstack([bert_embeddings.numpy(), meta_features])
y = relevance

print(f"Combined feature matrix shape: {X.shape} (768 BERT dims + 4 metadata dims)")

# Split at the REVIEW level (not the pair level) so no review appears in
# both train and test -- this avoids leaking pair information across splits.
X_train, X_test, y_train, y_test, votes_train, votes_test = train_test_split(
    X, y, helpful_votes, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# XGBRanker needs "group" boundaries: which rows are comparable to each
# other. We have no natural per-product grouping (median 1 review/product,
# see project notes), so we use ONE group spanning the whole train set --
# this is what makes this a *global* pairwise ranking formulation: any
# review can be compared against any other review, not just ones for the
# same product.
train_group = [len(X_train)]
test_group = [len(X_test)]

print("\nTraining XGBRanker (rank:pairwise, global group)...")

ranker = XGBRanker(
    objective="rank:pairwise",
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
)

ranker.fit(X_train, y_train, group=train_group)

# ----- Evaluation -----

test_scores = ranker.predict(X_test)

# NDCG over the whole test set treated as a single ranked list (since,
# again, there is no natural per-product grouping to evaluate within).
ndcg_full = ndcg_score(y_test.reshape(1, -1), test_scores.reshape(1, -1))
ndcg_at_10 = ndcg_score(y_test.reshape(1, -1), test_scores.reshape(1, -1), k=10)

# Pairwise accuracy: for randomly sampled test pairs with DIFFERENT true
# relevance, what fraction did the model rank in the correct order?
rng = np.random.default_rng(42)
n_pairs = 5000
idx_a = rng.integers(0, len(X_test), n_pairs)
idx_b = rng.integers(0, len(X_test), n_pairs)

valid = y_test[idx_a] != y_test[idx_b]
idx_a, idx_b = idx_a[valid], idx_b[valid]

true_order = y_test[idx_a] > y_test[idx_b]
pred_order = test_scores[idx_a] > test_scores[idx_b]
pairwise_accuracy = (true_order == pred_order).mean()

print("\n===== Helpfulness Ranking Results (global pairwise, BERT + metadata) =====")
print(f"Test samples:               {len(X_test)}")
print(f"Evaluable pairs sampled:    {len(idx_a)} (out of {n_pairs} sampled, ties dropped)")
print(f"NDCG (full test list):      {ndcg_full:.4f}")
print(f"NDCG@10:                    {ndcg_at_10:.4f}")
print(f"Pairwise ranking accuracy:  {pairwise_accuracy:.4f}")

print("\nFor reference, a random/untrained ranker's expected pairwise accuracy is ~0.50.")

# Persist the trained ranker so scoring.py (and later the FastAPI service)
# can load it directly instead of retraining on every request. XGBoost's
# native save_model format (JSON) is used rather than pickling, since it's
# the officially supported, version-stable way to persist XGBoost models.
os.makedirs("models", exist_ok=True)
ranker.save_model(MODEL_OUTPUT_PATH)
print(f"Saved trained helpfulness ranker to {MODEL_OUTPUT_PATH}")
