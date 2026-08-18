"""
Combined multimodal-review scoring module.

Loads the trained sentiment classifier (BERT-only logistic regression --
our ablation showed this beats BERT+CLIP fusion for sentiment) and the
trained helpfulness ranker (BERT embedding + metadata, XGBRanker), and
exposes score_review() to score a single new, raw review end to end.

This is deliberately kept as a plain importable module (not a script that
only runs top-to-bottom and exits) because the FastAPI service built in
the next stage will import score_review() directly rather than duplicating
this logic.
"""

import joblib
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from xgboost import XGBRanker

SENTIMENT_MODEL_PATH = "models/sentiment_bert_lr.joblib"
RANKER_MODEL_PATH = "models/helpfulness_ranker.json"
BERT_MODEL_NAME = "bert-base-uncased"

_device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading BERT tokenizer/model for on-the-fly text embedding...")
_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
_bert_model = BertModel.from_pretrained(BERT_MODEL_NAME).to(_device)
_bert_model.eval()

print("Loading trained sentiment classifier...")
_sentiment_classifier = joblib.load(SENTIMENT_MODEL_PATH)

print("Loading trained helpfulness ranker...")
_helpfulness_ranker = XGBRanker()
_helpfulness_ranker.load_model(RANKER_MODEL_PATH)

print("scoring.py ready.\n")


def _embed_text(text: str) -> np.ndarray:
    """
    Generate a BERT [CLS] embedding for one raw review string, using the
    exact same tokenization settings (max_length=128, truncation) as
    bert_embeddings.py used when building the training embeddings. Keeping
    these settings identical matters -- a mismatch here would silently
    shift the input distribution the trained models expect.
    """
    inputs = _tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    ).to(_device)
    with torch.no_grad():
        output = _bert_model(**inputs)
    return output.last_hidden_state[:, 0, :].cpu().numpy()  # shape (1, 768)


def score_review(text: str, rating: float, verified_purchase: bool) -> dict:
    """
    Score a single product review for predicted sentiment and predicted
    helpfulness.

    Parameters
    ----------
    text: the raw review text
    rating: star rating given with the review (1-5)
    verified_purchase: whether the purchase was verified

    Returns
    -------
    dict with:
      sentiment: predicted class label ("negative" / "neutral" / "positive")
      sentiment_confidence: predicted-class probabilities per label
      helpfulness_score: raw ranking score from the XGBoost ranker.
          NOTE: this is NOT a probability or a 0-1 scale -- it only has
          meaning relative to other reviews' scores (higher = predicted
          more helpful). Treat it as a ranking signal, not a percentage.
    """
    bert_embedding = _embed_text(text)  # (1, 768)

    # ----- Sentiment -----
    sentiment_label = _sentiment_classifier.predict(bert_embedding)[0]
    sentiment_probs = _sentiment_classifier.predict_proba(bert_embedding)[0]
    class_labels = _sentiment_classifier.classes_
    sentiment_confidence = {
        label: round(float(prob), 4) for label, prob in zip(class_labels, sentiment_probs)
    }

    # ----- Helpfulness -----
    # Must match the exact feature order used in helpfulness_ranking.py:
    # [768 BERT dims] + [text_length, rating, rating_extremity, verified_purchase]
    text_length = len(text.split())
    rating_extremity = abs(rating - 3)
    verified_flag = int(verified_purchase)

    meta_features = np.array([[text_length, rating, rating_extremity, verified_flag]])
    ranker_features = np.hstack([bert_embedding, meta_features])

    helpfulness_score = float(_helpfulness_ranker.predict(ranker_features)[0])

    return {
        "sentiment": sentiment_label,
        "sentiment_confidence": sentiment_confidence,
        "helpfulness_score": round(helpfulness_score, 4),
    }


if __name__ == "__main__":
    examples = [
        {
            "text": "This product broke after two days, complete waste of money, do not buy.",
            "rating": 1,
            "verified_purchase": True,
        },
        {
            "text": "It's fine. Does what it says. Nothing special.",
            "rating": 3,
            "verified_purchase": True,
        },
        {
            "text": "Absolutely love this! Great quality, fast shipping, exceeded expectations.",
            "rating": 5,
            "verified_purchase": True,
        },
    ]

    for example in examples:
        result = score_review(
            text=example["text"],
            rating=example["rating"],
            verified_purchase=example["verified_purchase"],
        )
        print(f"Review: {example['text']!r}")
        print(f"  -> {result}\n")
