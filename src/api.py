"""
FastAPI service exposing the multimodal review scoring system over HTTP.
 
Run from the project root with:
    python -m uvicorn src.api:app --reload
 
(Using `python -m uvicorn` -- not just `uvicorn` -- matters here: it
guarantees the project root gets added to Python's import path, which is
what lets `from src.scoring import score_review` resolve correctly.)
"""
import base64
import io
from typing import Optional

from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from src.scoring import score_review

app = FastAPI(title="Multimodal Review Scoring API", description="API for scoring reviews with multiple modalities", version="1.0")

# ----- Request / response schemas -----
# Pydantic models describe the EXACT shape of data going in and out.
# FastAPI uses these to: validate incoming requests automatically (reject
# bad input before your code even runs), generate interactive docs, and
# serialize your Python return value into correct JSON.

class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The text of the review to score")
    rating: float = Field(..., ge=1, le=5, description="The numeric star rating of the review (1-5)")
    verified_purchase: bool = Field(..., description="Whether the review is from a verified purchase")
    image_base64: Optional[str] = Field(
        None,
        description="Optional JPEG/PNG image as raw base64 or a data URL. Omitted for text-only scoring.",
    )


class ReviewResponse(BaseModel):
    sentiment: str
    sentiment_confidence: dict
    helpfulness_score: float
    image_processed: bool
    clip_embedding_dim: Optional[int] = None


def _decode_optional_image(image_base64: Optional[str]) -> Optional[Image.Image]:
    """Decode optional JSON image_base64 into a PIL Image. Empty/omitted -> None."""
    if image_base64 is None:
        return None

    payload = image_base64.strip()
    if not payload:
        return None

    if payload.lower().startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    payload += "=" * ((4 - len(payload) % 4) % 4)

    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image_base64: value is not valid base64.",
        )

    if not raw:
        raise HTTPException(
            status_code=400,
            detail="Invalid image_base64: decoded image data is empty.",
        )

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image_base64: value is not a readable image.",
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image_base64: value is not a readable image.",
        )

    return image


#----- Routes -----
@app.get("/")
def root():
    """Simple health-check endpoint to see if API is working"""
    return{"status": "ok", "message": "Multimodal Review Scoring API is running."}

@app.post("/score", response_model=ReviewResponse)
def score(review: ReviewRequest):
    """Score a review based on its text, rating, and verified purchase status.
    FastAPI has already validated 'review' against ReviewRequest by the time this function runs
    for example a review missing 'rating' or with rating=9 will give us error 422 before we even get here.
    Optional image_base64 is decoded here; CLIP runs only when an image is present."""
    try:
        image = _decode_optional_image(review.image_base64)
        result = score_review(review.text, review.rating, review.verified_purchase, image=image)
        return ReviewResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
