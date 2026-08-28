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

CLIP is optional and lazy-loaded: it runs only when an image is passed in,
and it never replaces BERT as the sentiment model or changes the ranker.
"""

import joblib
import numpy as np
import torch
from PIL import Image
from transformers import BertTokenizer, BertModel
from xgboost import XGBRanker

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SENTIMENT_MODEL_PATH = BASE_DIR / "models" / "sentiment_bert_lr.joblib"
RANKER_MODEL_PATH = BASE_DIR / "models" / "helpfulness_ranker.json"
BERT_MODEL_NAME = "bert-base-uncased"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

_device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading BERT tokenizer/model for on-the-fly text embedding...")
_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
_bert_model = BertModel.from_pretrained(BERT_MODEL_NAME).to(_device)
_bert_model.eval()

print("Loading trained sentiment classifier...")
if not SENTIMENT_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Sentiment model file not found at: {SENTIMENT_MODEL_PATH}. "
        "Ensure model artifacts exist in models/ directory."
    )
_sentiment_classifier = joblib.load(SENTIMENT_MODEL_PATH)

print("Loading trained helpfulness ranker...")
if not RANKER_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Helpfulness ranker model file not found at: {RANKER_MODEL_PATH}. "
        "Ensure model artifacts exist in models/ directory."
    )
_helpfulness_ranker = XGBRanker()
_helpfulness_ranker.load_model(str(RANKER_MODEL_PATH))

print("scoring.py ready.\n")

# Loaded only on the first request that includes an image.
_clip_processor = None
_clip_model = None


def _get_clip():
    """Load CLIP processor/model once, on first image request."""
    global _clip_processor, _clip_model
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel

        print("Loading CLIP processor/model for on-the-fly image embedding...")
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(_device)
        _clip_model.eval()
    return _clip_processor, _clip_model


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


def _embed_image(image: Image.Image) -> np.ndarray:
    """
    Generate a CLIP image embedding using the same model and preprocessing
    as clip_embedding.py: openai/clip-vit-base-patch32, RGB conversion,
    CLIPProcessor + CLIPModel.get_image_features, no L2 normalization.
    """
    processor, model = _get_clip()
    rgb_image = image.convert("RGB")
    inputs = processor(images=rgb_image, return_tensors="pt").to(_device)
    with torch.no_grad():
        output = model.get_image_features(**inputs)
        # clip_embedding.py: get_image_features may return a tensor (512-d)
        # or a pooled vision output. Use pooler_output when present, then
        # apply visual_projection if needed so the vector stays 512-d like
        # the training embeddings (hidden size is 768 before projection).
        if hasattr(output, "pooler_output"):
            image_embedding = output.pooler_output
        else:
            image_embedding = output
        if image_embedding.shape[-1] != 512 and hasattr(model, "visual_projection"):
            image_embedding = model.visual_projection(image_embedding)
    return image_embedding.cpu().numpy()


def score_review(text: str, rating: float, verified_purchase: bool, image=None) -> dict:
    """
    Score a single product review for predicted sentiment and predicted
    helpfulness.

    Parameters
    ----------
    text: the raw review text
    rating: star rating given with the review (1-5)
    verified_purchase: whether the purchase was verified
    image: optional PIL Image; when provided, CLIP is run to confirm the
        image was processed. CLIP does not affect sentiment or helpfulness.

    Returns
    -------
    dict with:
      sentiment: predicted class label ("negative" / "neutral" / "positive")
      sentiment_confidence: predicted-class probabilities per label
      helpfulness_score: raw ranking score from the XGBoost ranker.
          NOTE: this is NOT a probability or a 0-1 scale -- it only has
          meaning relative to other reviews' scores (higher = predicted
          more helpful). Treat it as a ranking signal, not a percentage.
      image_processed: True only if CLIP ran and produced a finite (1, 512)
          embedding
      clip_embedding_dim: 512 if an image was processed, otherwise None
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

    image_processed = False
    clip_embedding_dim = None
    if image is not None:
        clip_embedding = _embed_image(image)
        if clip_embedding.shape != (1, 512):
            raise ValueError(
                f"CLIP embedding had unexpected shape {clip_embedding.shape}; expected (1, 512)."
            )
        if not np.isfinite(clip_embedding).all():
            raise ValueError("CLIP embedding contained non-finite values.")
        image_processed = True
        clip_embedding_dim = int(clip_embedding.shape[1])

    return {
        "sentiment": sentiment_label,
        "sentiment_confidence": sentiment_confidence,
        "helpfulness_score": round(helpfulness_score, 4),
        "image_processed": image_processed,
        "clip_embedding_dim": clip_embedding_dim,
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
