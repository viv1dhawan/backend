import uuid
import sqlalchemy
from datetime import datetime
# Import metadata from the local database module
from database import metadata

# Table definition for users
users = sqlalchemy.Table(
    "users",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("email", sqlalchemy.String(255), unique=True, nullable=False),
    sqlalchemy.Column("hashed_password", sqlalchemy.String(255), nullable=False),
    sqlalchemy.Column("first_name", sqlalchemy.String(255)),
    sqlalchemy.Column("last_name", sqlalchemy.String(255)),
    sqlalchemy.Column("is_verified", sqlalchemy.Boolean, default=False, nullable=False),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow, nullable=False),
    sqlalchemy.Column("updated_at", sqlalchemy.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False),
)

# Table definition for password reset tokens
password_reset_tokens = sqlalchemy.Table(
    "password_reset_tokens",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("email", sqlalchemy.String(255), nullable=False),
    sqlalchemy.Column("token", sqlalchemy.String(255), unique=True, nullable=False),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow, nullable=False),
    sqlalchemy.Column("expires_at", sqlalchemy.DateTime, nullable=False),
)

# Table definition for email verification tokens
email_verification_tokens = sqlalchemy.Table(
    "email_verification_tokens",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("email", sqlalchemy.String(255), nullable=False),
    sqlalchemy.Column("token", sqlalchemy.String(255), unique=True, nullable=False),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow, nullable=False),
    sqlalchemy.Column("expires_at", sqlalchemy.DateTime, nullable=False),
)

# Table definition for gravity_data
gravity_data = sqlalchemy.Table(
    "gravity_data",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("latitude", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("longitude", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("elevation", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("gravity", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("bouguer", sqlalchemy.Float, nullable=True),  # Calculated Bouguer anomaly
    sqlalchemy.Column("cluster", sqlalchemy.Integer, nullable=True),  # K-Means cluster assignment
    sqlalchemy.Column("anomaly", sqlalchemy.Integer, nullable=True),  # Isolation Forest anomaly (-1 or 1)
    sqlalchemy.Column("distance_km", sqlalchemy.Float, nullable=True), # Distance from a reference point
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.now),
)

# Table definition for earthquakes
earthquakes = sqlalchemy.Table(
    "earthquakes",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.String(50), primary_key=True),
    sqlalchemy.Column("time", sqlalchemy.DateTime, nullable=False),
    sqlalchemy.Column("latitude", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("longitude", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("depth", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("mag", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("place", sqlalchemy.String(255)),
    sqlalchemy.Column("updated_at", sqlalchemy.DateTime, default=datetime.now, onupdate=datetime.now),
)

# Table definition for questions in Q&A forum
questions = sqlalchemy.Table(
    "questions",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id"), nullable=False), # Link to user who asked
    sqlalchemy.Column("text", sqlalchemy.String(1000), nullable=False),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.now, nullable=False),
)

# Table definition for comments
comments = sqlalchemy.Table(
    "comments",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
    sqlalchemy.Column("text", sqlalchemy.String(500), nullable=False),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.now, nullable=False),
    sqlalchemy.Column("question_id", sqlalchemy.String(36), sqlalchemy.ForeignKey("questions.id"), nullable=False),
)

# Table definition for question_likes
question_likes = sqlalchemy.Table(
    "question_likes",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id"), nullable=False),
    sqlalchemy.Column("question_id", sqlalchemy.String(36), sqlalchemy.ForeignKey("questions.id"), nullable=False),
    sqlalchemy.Column("type", sqlalchemy.String(10), nullable=False), # 'like' or 'dislike'
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.now, nullable=False),
    # Unique constraint to ensure a user can only like/dislike a question once
    sqlalchemy.UniqueConstraint("user_id", "question_id", name="uq_user_question_interaction")
)

# New table definition for researchers
researchers = sqlalchemy.Table(
    "researchers",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("title", sqlalchemy.String(255), nullable=False),
    sqlalchemy.Column("authors", sqlalchemy.String(500), nullable=False),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id"), nullable=True),
    sqlalchemy.Column("profile", sqlalchemy.String(500), nullable=False),
    sqlalchemy.Column("publication_date", sqlalchemy.String(50), nullable=False), # Storing as string to match schema
    sqlalchemy.Column("url", sqlalchemy.String(1000), nullable=False),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.now, nullable=False),
    sqlalchemy.Column("updated_at", sqlalchemy.DateTime, default=datetime.now, onupdate=datetime.now, nullable=False),
)
