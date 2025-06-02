import sqlalchemy
from datetime import datetime
# Import metadata from the local database module (now a sibling)
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

# Table definition for earthquake data
earthquakes = sqlalchemy.Table(
    "earthquakes",
    metadata,
    sqlalchemy.Column("time", sqlalchemy.DateTime, nullable=False),
    sqlalchemy.Column("latitude", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("longitude", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("depth", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("mag", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("magtype", sqlalchemy.String(50), nullable=False),
    sqlalchemy.Column("nst", sqlalchemy.Integer, nullable=False),
    sqlalchemy.Column("gap", sqlalchemy.Integer, nullable=False),
    sqlalchemy.Column("dmin", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("rms", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("net", sqlalchemy.String(50), nullable=False),
    sqlalchemy.Column("id", sqlalchemy.String(100), primary_key=True),  # Primary key here
    sqlalchemy.Column("updated", sqlalchemy.DateTime),
    sqlalchemy.Column("place", sqlalchemy.String(255)),
    sqlalchemy.Column("type", sqlalchemy.String(50)),
    sqlalchemy.Column("horizontalError", sqlalchemy.Float,nullable=False),
    sqlalchemy.Column("depthError", sqlalchemy.Float,nullable=False),
    sqlalchemy.Column("magError", sqlalchemy.Float,nullable=False),
    sqlalchemy.Column("magNst", sqlalchemy.Integer,nullable=False),
    sqlalchemy.Column("status", sqlalchemy.String(50)),
    sqlalchemy.Column("locationSource", sqlalchemy.String(50)),
    sqlalchemy.Column("magSource", sqlalchemy.String(50)),
)

# Table definition for gravity data
gravity_data = sqlalchemy.Table(
    "gravity_data",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("latitude", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("longitude", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("elevation", sqlalchemy.Float),
    sqlalchemy.Column("gravity", sqlalchemy.Float),
    # Add columns for processed data if you want to persist them directly in the DB
    sqlalchemy.Column("bouguer", sqlalchemy.Float, nullable=True),
    sqlalchemy.Column("cluster", sqlalchemy.Integer, nullable=True),
    sqlalchemy.Column("anomaly", sqlalchemy.Integer, nullable=True),
    sqlalchemy.Column("distance_km", sqlalchemy.Float, nullable=True),
)
