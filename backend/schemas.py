"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel
from typing import List


class Detection(BaseModel):
    """Single detection result."""
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float


class DetectionResponse(BaseModel):
    """Response containing list of detections."""
    detections: List[Detection]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str


class ClassesResponse(BaseModel):
    """Model classes response."""
    classes: dict

