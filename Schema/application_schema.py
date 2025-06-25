from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
from typing import List, Optional, Dict, Any

class Researcher(BaseModel):
    title: str
    authors: str
    publication_date: str
    url: str


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

class EarthquakeQuery(BaseModel):
    start_date: datetime
    end_date: datetime
    min_mag: Optional[float] = None
    max_mag: Optional[float] = None
    min_depth: Optional[float] = None
    max_depth: Optional[float] = None

# --- Enums ---

# --- Enums ---

class LikeDislikeType(str, Enum):
    """Enum for the type of interaction: 'like' or 'dislike'."""
    LIKE = "like"
    DISLIKE = "dislike"

# --- Request Models ---

class QuestionCreate(BaseModel):
    """
    Schema for creating a new question.
    """
    text: str = Field(..., min_length=1, max_length=1000, description="The text content of the question.")

class CommentCreate(BaseModel):
    """
    Schema for creating a new comment on a question.
    """
    text: str = Field(..., min_length=1, max_length=500, description="The text of the comment.")

class QuestionInteractionCreate(BaseModel):
    """
    Schema for creating a new like or dislike interaction on a question.
    NOTE: The backend POST endpoint for interaction uses query parameters for 'type',
    so this schema is likely for internal CRUD operations if you're passing a dict.
    The user_id is derived from current_user in the API.
    """
    type: LikeDislikeType = Field(..., description="Type of interaction: 'like' or 'dislike'.")
    user_id: str = Field(..., description="The ID of the user performing the interaction.")
    # Changed user_id to str based on common UUID usage in modern apps, adjust if it's int in your DB

# --- Response Models ---

class CommentResponse(BaseModel):
    """
    Schema for returning a comment.
    """
    id: str
    text: str
    created_at: datetime # Timezone-naive datetime (consider using timezone-aware if possible)
    # user_id: str # You might want to add the user_id here if displaying who made the comment

    # Pydantic v2 configuration
    model_config = ConfigDict(from_attributes=True)
    # Pydantic v1 equivalent:
    # class Config:
    #     orm_mode = True

class QuestionInteractionResponse(BaseModel):
    """
    Schema for returning a question interaction (like/dislike).
    This is the model that was crucial for the backend fix.
    """
    id: str
    user_id: str # Changed to str consistent with typical UUIDs for user IDs
    question_id: str
    type: str # Use str here as the 'removed' status is not part of the enum (or consider LikeDislikeType)
    created_at: datetime # Timezone-naive datetime
    updated_at: datetime # Interactions often have an updated_at timestamp

    # Pydantic v2 configuration
    model_config = ConfigDict(from_attributes=True)
    # Pydantic v1 equivalent:
    # class Config:
    #     orm_mode = True

class QuestionResponse(BaseModel):
    """
    Schema for returning a question, including its comments and like/dislike counts.
    """
    id: str
    text: str
    created_at: datetime # Timezone-naive datetime
    comments: List[CommentResponse] = [] # List of CommentResponse
    likes_count: int = 0
    dislikes_count: int = 0

    # Pydantic v2 configuration
    model_config = ConfigDict(from_attributes=True)
    # Pydantic v1 equivalent:
    # class Config:
    #     orm_mode = True
