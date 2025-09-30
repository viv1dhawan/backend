from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
from typing import List, Optional

# ============================================================
# Enums
# ============================================================

class LikeDislikeType(str, Enum):
    like = "like"
    dislike = "dislike"


# ============================================================
# Input / Request Models
# ============================================================

# --- Researcher Requests ---
class ResearcherInputRequest(BaseModel):
    researcher_id: int = Field(..., description="The ID of the researcher to retrieve.")

class ResearcherGetRequest(BaseModel):
    researcher_id: int = Field(..., description="The ID of the researcher to retrieve.")

class ResearcherUpdateRequest(BaseModel):
    researcher_id: int = Field(..., description="The ID of the researcher to update.")
    title: Optional[str] = None
    authors: Optional[str] = None
    profile: Optional[str] = None
    publication_date: Optional[str] = None
    url: Optional[str] = None
    user_id: Optional[int] = None

class ResearcherAddRequest(BaseModel):
    title: str
    authors: str
    user_id: Optional[int] = None
    profile: str
    publication_date: str
    url: str

class ResearcherDeleteRequest(BaseModel):
    researcher_id: int = Field(..., description="The ID of the researcher to delete.")


# --- Question & Comment Requests ---
class QuestionInputRequest(BaseModel):
    question_id: str = Field(..., description="The ID of the question to delete.")

class QuestionCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="The text content of the question.")

class CommentAddRequest(BaseModel):
    question_id: str = Field(..., description="The ID of the question to comment on.")
    text: str = Field(..., min_length=1, max_length=500, description="The text of the comment.")

class QuestionInteractRequest(BaseModel):
    type: LikeDislikeType
    question_id: str
    
class QuestionInteractionDeleteRequest(BaseModel):
    question_id: str = Field(..., description="The ID of the question the interaction belongs to.")
    interaction_id: str = Field(..., description="The ID of the interaction to delete.")


# --- Hugging Face Requests ---
class HuggingFaceRequest(BaseModel):
    prompt: str


# ============================================================
# Response Models
# ============================================================

class ResearcherResponse(BaseModel):
    """Schema for returning a researcher entity."""
    id: Optional[int] = None
    user_id: Optional[int] = None
    title: str
    authors: str
    publication_date: str
    url: str
    profile: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class HuggingFaceResponse(BaseModel):
    generated_text: str

class CommentResponse(BaseModel):
    id: int
    text: str
    created_at: datetime

class QuestionInteractionResponse(BaseModel):
    id: int
    user_id: int
    question_id: str
    type: str
    updated_at: datetime

class QuestionResponse(BaseModel):
    id: str
    text: str
    created_at: datetime
    comments: list[CommentResponse] = []
    likes_count: int = 0
    dislikes_count: int = 0

class QuestionsListResponse(BaseModel):
    questions: list[QuestionResponse] = []