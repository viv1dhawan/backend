from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
from typing import List, Optional, Dict, Any

class Researcher(BaseModel):
    # Added user_id to the Researcher schema
    id: Optional[int] = None # For retrieval, ID will be present
    user_id: Optional[int] = None # For retrieval, user_id will be present
    title: str
    authors: str
    publication_date: str
    url: str
    profile : str
    created_at: Optional[datetime] = None # Optional for creation, auto-set by DB
    updated_at: Optional[datetime] = None # Optional for creation, auto-set by DB


# Pydantic model for a single raw gravity data point
class GravityDataPoint(BaseModel):
    latitude: float
    longitude: float
    elevation: float
    gravity: float

# Pydantic model for processed gravity data, inheriting from GravityDataPoint
class ProcessedGravityData(GravityDataPoint):
    id: Optional[int] = None # Optional ID for existing records
    bouguer: Optional[float] = None
    cluster: Optional[int] = None
    anomaly: Optional[int] = None # -1 for anomaly, 1 for normal
    distance_km: Optional[float] = None

    class Config:
        from_attributes = True # Allows mapping from SQLAlchemy models

# Pydantic model for the response after uploading a CSV file
class UploadResponse(BaseModel):
    message: str
    row_count: int

# Pydantic model for anomaly detection results
class AnomalyDetectionResult(BaseModel):
    latitude: float
    longitude: float
    elevation: float
    gravity: float
    anomaly: int # -1 for anomaly, 1 for normal

    class Config:
        from_attributes = True

# Pydantic model for K-Means clustering results
class ClusteringResult(BaseModel):
    latitude: float
    longitude: float
    elevation: float
    gravity: float
    cluster: int

    class Config:
        from_attributes = True

# Pydantic model for Plotly graph data (to be returned as JSON)
class PlotlyGraph(BaseModel):
    data: List[Dict[str, Any]]
    layout: Dict[str, Any]

# Pydantic model for error responses
class ErrorResponse(BaseModel):
    detail: str

# Existing EarthquakeQuery, now intended for POST request body
class EarthquakeQuery(BaseModel):
    start_date: datetime
    end_date: datetime
    min_mag: Optional[float] = None
    max_mag: Optional[float] = None
    min_depth: Optional[float] = None
    max_depth: Optional[float] = None

# --- Enums ---

class LikeDislikeType(str, Enum):
    """Enum for the type of interaction: 'like' or 'dislike'."""
    LIKE = "like"
    DISLIKE = "dislike"

# --- Request Models for JSON Input ---

# For /app_router/kmeans-clusters/
class KMeansRequest(BaseModel):
    n_clusters: int = Field(3, ge=1, description="Number of clusters for K-Means.")

# For /app_router/anomaly-detection/
class AnomalyDetectionRequest(BaseModel):
    contamination: float = Field(0.05, gt=0, lt=0.5, description="Contamination parameter for Isolation Forest (0 to 0.5).")

# For /qna_router/comments/ (now with question_id in body)
class CommentAddRequest(BaseModel):
    question_id: str = Field(..., description="The ID of the question to comment on.")
    text: str = Field(..., min_length=1, max_length=500, description="The text of the comment.")

# For /qna_router/interact/ (now with question_id and type in body)
class QuestionInteractRequest(BaseModel):
    question_id: str = Field(..., description="The ID of the question to interact with.")
    interaction_type: LikeDislikeType = Field(..., description="Type of interaction: 'like' or 'dislike'.")

# For /qna_router/interaction-delete/ (new endpoint for deleting interactions by JSON)
class QuestionInteractionDeleteRequest(BaseModel):
    question_id: str = Field(..., description="The ID of the question the interaction belongs to.")
    interaction_id: str = Field(..., description="The ID of the interaction to delete.")

# For /researchers/get-by-id/ (new endpoint for getting researcher by ID via JSON)
class ResearcherGetRequest(BaseModel):
    researcher_id: int = Field(..., description="The ID of the researcher to retrieve.") # Changed to int

# For /researchers/update-researcher/ (new endpoint for updating researcher via JSON)
class ResearcherUpdateRequest(BaseModel):
    researcher_id: int = Field(..., description="The ID of the researcher to update.") # Changed to int
    title: Optional[str] = None
    authors: Optional[str] = None
    profile: Optional[str] = None
    publication_date: Optional[str] = None
    url: Optional[str] = None
    user_id:Optional[int] = None

# New Pydantic models for Together AI integration
class TogetherAIRequest(BaseModel):
    """Schema for a request to the Together AI endpoint."""
    prompt: str = Field(..., min_length=1, description="The text prompt for the AI model.")

class TogetherAIResponse(BaseModel):
    """Schema for the response from the Together AI endpoint."""
    generated_text: str = Field(..., description="The text generated by the Together AI model.")


# Existing
class QuestionCreate(BaseModel):
    """
    Schema for creating a new question.
    """
    text: str = Field(..., min_length=1, max_length=1000, description="The text content of the question.")

# Existing
class ResearcherDeleteRequest(BaseModel):
    researcher_id: int = Field(..., description="The ID of the researcher to delete.") # Changed to int


# --- Response Models ---

class CommentResponse(BaseModel):
    """
    Schema for returning a comment.
    """
    id: str
    text: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class QuestionInteractionResponse(BaseModel):
    """
    Schema for returning a question interaction (like/dislike).
    """
    id: str
    user_id: int
    question_id: str
    type: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class QuestionResponse(BaseModel):
    """
    Schema for returning a question, including its comments and like/dislike counts.
    """
    id: str
    text: str
    created_at: datetime
    comments: List[CommentResponse] = []
    likes_count: int = 0
    dislikes_count: int = 0

    model_config = ConfigDict(from_attributes=True)

