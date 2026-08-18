import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from xgboost import XGBRanker


ALIGNED_PATH = "data/processed/aligned_embeddings.pt"
CSV_PATH = "data/processed/multimodal_reviews.csv"
OUTPUT_PATH = "data/processed/helpfulness_ablation_summary.csv"

# Same fixed relevance cutoffs used in helpfulness_ranking.py
RELEVANCE_BINS = [-1, 0, 2, 5, 10, 10**9]
RELEVANCE_LABELS = [0, 1, 2, 3, 4]


def helpful_vote_to_relevance(vote_series):
    return pd.cut(vote_series, bins=RELEVANCE_BINS, labels=RELEVANCE_LABELS).astype(int)


def run_ranking_experiment(name, X, y, rng_seed=42):
    """Train + evaluate one feature configuration with the exact same
    split, ranker settings, and metrics as helpfulness_ranking.py, so
    results across configurations are directly comparable."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rng_seed, stratify=y
    )

    ranker = XGBRanker(
        objective="rank:pairwise",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=rng_seed,
    )
    ranker.fit(X_train, y_train, group=[len(X_train)])

    test_scores = ranker.predict(X_test)

    ndcg_full = ndcg_score(y_test.reshape(1, -1), test_scores.reshape(1, -1))
    ndcg_at_10 = ndcg_score(y_test.reshape(1, -1), test_scores.reshape(1, -1), k=10)

    rng = np.random.default_rng(rng_seed)
    n_pairs = 5000
    idx_a = rng.integers(0, len(X_test), n_pairs)
    idx_b = rng.integers(0, len(X_test), n_pairs)
    valid = y_test[idx_a] != y_test[idx_b]
    idx_a, idx_b = idx_a[valid], idx_b[valid]
    true_order = y_test[idx_a] > y_test[idx_b]
    pred_order = test_scores[idx_a] > test_scores[idx_b]
    pairwise_accuracy = (true_order == pred_order).mean()

    return {
        "Experiment": name,
        "Feature dims": X.shape[1],
        "NDCG (full)": ndcg_full,
        "NDCG@10": ndcg_at_10,
        "Pairwise accuracy": pairwise_accuracy,
    }


print("Loading aligned embeddings (BERT + CLIP)...")
aligned_data = torch.load(ALIGNED_PATH, weights_only=False)
bert_embeddings = aligned_data["bert_embeddings"].numpy()
clip_embeddings = aligned_data["clip_embeddings"].numpy()
original_indices = aligned_data["original_indices"]

df = pd.read_csv(CSV_PATH)
helpful_votes = df.iloc[original_indices]["helpful_vote"].values
relevance = helpful_vote_to_relevance(pd.Series(helpful_votes)).values

review_text = df.iloc[original_indices]["text"].fillna("")
text_length = review_text.apply(lambda t: len(str(t).split())).values
ratings = df.iloc[original_indices]["rating"].values
rating_extremity = np.abs(ratings - 3)
verified_purchase = (
    df.iloc[original_indices]["verified_purchase"]
    .astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0).values
)
meta_features = np.column_stack([text_length, ratings, rating_extremity, verified_purchase])

y = relevance

configs = {
    "BERT + metadata (current production model)": np.hstack([bert_embeddings, meta_features]),
    "CLIP + metadata (image only)": np.hstack([clip_embeddings, meta_features]),
    "BERT + CLIP + metadata (full fusion)": np.hstack(
        [bert_embeddings, clip_embeddings, meta_features]
    ),
}

results = []
for name, X in configs.items():
    print(f"\nTraining: {name}  (feature shape {X.shape})...")
    results.append(run_ranking_experiment(name, X, y))

summary_df = pd.DataFrame(results).set_index("Experiment")

print("\n===== Helpfulness Ranking Ablation Summary =====\n")
print(summary_df.round(4).to_string())

summary_df.round(4).to_csv(OUTPUT_PATH)
print(f"\nSaved summary table to {OUTPUT_PATH}")
