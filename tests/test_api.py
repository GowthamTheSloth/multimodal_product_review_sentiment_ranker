"""API endpoint integration tests using FastAPI TestClient."""
import base64
import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def _generate_test_image_base64(format: str = "PNG", data_url: bool = False) -> str:
    """Helper to generate a small base64 test image."""
    img = Image.new("RGB", (16, 16), color=(255, 100, 50))
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    raw_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    if data_url:
        return f"data:image/{format.lower()};base64,{raw_b64}"
    return raw_b64


def test_health_check_root():
    """Verify GET / returns 200 and healthy status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "Multimodal Review Scoring API" in data["message"]


def test_score_valid_text_only():
    """Verify POST /score succeeds for text-only review request."""
    payload = {
        "text": "Excellent quality, sturdy build, and arrived promptly!",
        "rating": 5.0,
        "verified_purchase": True,
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["positive", "neutral", "negative"]
    assert isinstance(data["sentiment_confidence"], dict)
    assert len(data["sentiment_confidence"]) == 3
    assert isinstance(data["helpfulness_score"], float)
    assert data["image_processed"] is False
    assert data["clip_embedding_dim"] is None


def test_score_valid_with_base64_image():
    """Verify POST /score processes valid base64 image."""
    payload = {
        "text": "Looks just like the picture, very satisfied with this order.",
        "rating": 4.5,
        "verified_purchase": True,
        "image_base64": _generate_test_image_base64(format="PNG", data_url=False),
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["positive", "neutral", "negative"]
    assert isinstance(data["helpfulness_score"], float)
    assert data["image_processed"] is True
    assert data["clip_embedding_dim"] == 512


def test_score_valid_with_data_url_image():
    """Verify POST /score accepts data URL formatted base64."""
    payload = {
        "text": "Great color representation in person!",
        "rating": 5.0,
        "verified_purchase": True,
        "image_base64": _generate_test_image_base64(format="JPEG", data_url=True),
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["image_processed"] is True
    assert data["clip_embedding_dim"] == 512


def test_score_empty_or_whitespace_image_handled():
    """Verify empty string or whitespace in image_base64 is treated as text-only."""
    payload = {
        "text": "Good standard product, nothing unusual.",
        "rating": 3.0,
        "verified_purchase": True,
        "image_base64": "   ",
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["image_processed"] is False
    assert data["clip_embedding_dim"] is None


def test_score_invalid_rating_too_low():
    """Verify rating below 1 returns 422 validation error."""
    payload = {
        "text": "Awful product.",
        "rating": 0.5,
        "verified_purchase": True,
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 422


def test_score_invalid_rating_too_high():
    """Verify rating above 5 returns 422 validation error."""
    payload = {
        "text": "Superb product.",
        "rating": 6.0,
        "verified_purchase": True,
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 422


def test_score_empty_text():
    """Verify empty review text returns 422 validation error."""
    payload = {
        "text": "",
        "rating": 4.0,
        "verified_purchase": True,
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 422


def test_score_missing_required_fields():
    """Verify missing verified_purchase field returns 422 validation error."""
    payload = {
        "text": "Missing verified purchase flag.",
        "rating": 3.0,
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 422


def test_score_invalid_image_base64_syntax():
    """Verify malformed base64 payload returns 400 Bad Request."""
    payload = {
        "text": "Has invalid base64 image data.",
        "rating": 2.0,
        "verified_purchase": True,
        "image_base64": "!!!not_valid_base64???",
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 400
    assert "Invalid image_base64" in response.json()["detail"]


def test_score_corrupted_image_bytes():
    """Verify valid base64 encoding non-image bytes returns 400 Bad Request."""
    non_image_b64 = base64.b64encode(b"This is plain text and definitely not a PNG or JPEG file.").decode("utf-8")
    payload = {
        "text": "Has corrupted non-image bytes.",
        "rating": 2.0,
        "verified_purchase": True,
        "image_base64": non_image_b64,
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 400
    assert "not a readable image" in response.json()["detail"]
