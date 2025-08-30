# models.py
from pydantic import BaseModel, Field
from typing import Dict, Optional

class EmailRequest(BaseModel):
    subject: str = Field(default="", description="Email subject line")
    body: str = Field(default="", description="Email body content")
    return_probabilities: bool = Field(default=False, description="Return all category probabilities")

class EmailResponse(BaseModel):
    category: str = Field(description="Predicted email category")
    confidence: float = Field(description="Confidence score (0-1)")
    text_preview: str = Field(description="Preview of processed text")
    method: Optional[str] = Field(default=None, description="Classification method used")
    all_probabilities: Optional[Dict[str, float]] = Field(default=None, description="All category probabilities")