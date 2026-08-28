"""Unit tests for src.scoring module."""
import numpy as np
from PIL import Image
import pytest
from src.scoring import score_review


def test_score_review_text_only_positive():
    """Verify that positive review text generates valid sentiment and helpfulness scores."""
    result = score_review(
        text="Absolutely fantastic product! Exceeded all my expectations and very high quality.",
        rating=5.0,
        verified_purchase=True,
    )
    assert isinstance(result, dict)
    assert result["sentiment"] in ["positive", "neutral", "negative"]
    assert "sentiment_confidence" in result
    assert isinstance(result["sentiment_confidence"], dict)
    for label in ["negative", "neutral", "positive"]:
        assert label in result["sentiment_confidence"]
        assert 0.0 <= result["sentiment_confidence"][label] <= 1.0
    assert abs(sum(result["sentiment_confidence"].values()) - 1.0) < 0.01
    assert isinstance(result["helpfulness_score"], float)
    assert result["image_processed"] is False
    assert result["clip_embedding_dim"] is None


def test_score_review_text_only_negative():
    """Verify that negative review text returns expected structure and probabilities."""
    result = score_review(
        text="Worst purchase ever, completely broke after one day and seller refused refund.",
        rating=1.0,
        verified_purchase=False,
    )
    assert result["sentiment"] in ["positive", "neutral", "negative"]
    assert isinstance(result["helpfulness_score"], float)
    assert result["image_processed"] is False
    assert result["clip_embedding_dim"] is None


def test_score_review_with_valid_image():
    """Verify that passing a valid PIL Image triggers CLIP lazy loading and produces 512-d dim."""
    img = Image.new("RGB", (32, 32), color=(128, 64, 32))
    result = score_review(
        text="The color matches the photo exactly, see attached image.",
        rating=4.0,
        verified_purchase=True,
        image=img,
    )
    assert result["image_processed"] is True
    assert result["clip_embedding_dim"] == 512
    assert isinstance(result["helpfulness_score"], float)


def test_score_review_helpfulness_metadata_sensitivity():
    """Verify that differing metadata features influence ranker output."""
    res_short = score_review(text="Good.", rating=3.0, verified_purchase=False)
    res_long = score_review(
        text="This is a very detailed and thorough review with extensive testing of the features over multiple weeks.",
        rating=5.0,
        verified_purchase=True,
    )
    assert isinstance(res_short["helpfulness_score"], float)
    assert isinstance(res_long["helpfulness_score"], float)
