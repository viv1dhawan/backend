# app.py (Extended with Q&A Endpoints)
from fastapi import APIRouter, HTTPException, Depends, Form, File, UploadFile, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy.interpolate import griddata
from math import radians, cos, sin, asin, sqrt
import json
import os

# Import database operations
import db_info.user_db as user_db
import db_info.application_db as application_db

# Import Pydantic schemas
from Schema.user_schema import UserCreate, UserOut, PasswordReset, PasswordResetRequest, EmailVerificationRequest, EmailVerification, UserUpdate
from Schema.application_schema import EarthquakeQuery, GravityDataPoint, ProcessedGravityData, UploadResponse, AnomalyDetectionResult, ClusteringResult, PlotlyGraph, ErrorResponse, Researcher
# Import Q&A related schemas
from Schema.application_schema import QuestionCreate, QuestionResponse, CommentCreate, CommentResponse, LikeDislikeType, QuestionInteractionResponse

# Import database session factory (assuming it's defined in database.py)
from database import get_db_session # Only get_db_session is needed now
from sqlalchemy.ext.asyncio import AsyncSession # Crucial for type hinting async sessions

# Define API Routers
users_router = APIRouter()
app_router = APIRouter()
qna_router = APIRouter() # New router for Q&A features
researcher_router = APIRouter() # New router for Researcher features

# --- Security Configuration ---
SECRET_KEY = "IAMVIVEKDHAWAN_SUPER_SECRET_KEY"  # IMPORTANT: Use environment variables in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/token")

# In-memory token blacklist (Use Redis or database in production for persistence)
TOKEN_BLACKLIST = set()

# --- Utility Functions for Authentication ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Creates a JWT access token.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_db_session) # Inject session here
) -> Dict[str, Any]:
    """
    Authenticates and retrieves the current user based on the provided JWT token.
    """
    if token in TOKEN_BLACKLIST:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has been revoked")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        current_user = await user_db.get_user_by_email(session, email) # Pass session
        if not current_user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        return current_user
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

# --- User Management Endpoints (users_router) ---

@users_router.post("/signup", response_model=UserOut, summary="Register a new user")
async def signup(
    user: UserCreate,
    session: AsyncSession = Depends(get_db_session) # Inject session
):
    """
    Registers a new user with first name, last name, email, and password.
    Hashes the password and sets is_verified to False.
    """
    existing_user = await user_db.get_user_by_email(session, user.email) # Pass session
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    created_user_data = await user_db.create_user(session, user.model_dump()) # Pass session
    return UserOut(**created_user_data)

@users_router.post("/token", summary="Authenticate user and get access token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_db_session) # Inject session
):
    """
    Authenticates a user with username (email) and password,
    and returns an access token upon successful login.
    """
    current_user = await user_db.get_user_by_email(session, form_data.username) # Pass session
    if not current_user or not user_db.verify_password(form_data.password, current_user["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": current_user["email"]})
    return {"access_token": access_token, "token_type": "bearer"}

@users_router.post("/password-reset-request", summary="Request a password reset token")
async def password_reset_request(
    request: PasswordResetRequest,
    session: AsyncSession = Depends(get_db_session) # Inject session
):
    """
    Requests a password reset token for a given email.
    A token will be generated and (simulated) sent to the user's email.
    """
    current_user = await user_db.get_user_by_email(session, request.email) # Pass session
    if not current_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    token = await user_db.create_password_reset_token(session, current_user["email"]) # Pass session
    # In a real application, you would send this token via email
    print(f"Password reset token for {request.email}: {token}")  # For testing/development
    return {"message": "Password reset token generated and (simulated) sent to email.", "token": token}

@users_router.post("/password-reset", summary="Reset user password with token")
async def password_reset(
    password_reset_data: PasswordReset,
    session: AsyncSession = Depends(get_db_session) # Inject session
):
    """
    Resets a user's password using the provided token and new password.
    """
    email = await user_db.verify_password_reset_token(session, password_reset_data.token) # Pass session
    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired token")
    await user_db.update_password(session, email, password_reset_data.new_password) # Pass session
    return {"message": "Password updated successfully"}

@users_router.post("/request-email-verification/", summary="Request email verification token")
async def request_email_verification(
    request: EmailVerificationRequest,
    session: AsyncSession = Depends(get_db_session) # Inject session
):
    """
    Requests an email verification token for a given email.
    In a real app, this would send an email with the token.
    """
    current_user = await user_db.get_user_by_email(session, request.email) # Pass session
    if not current_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if current_user["is_verified"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already verified")

    token = await user_db.generate_verification_token(session, request.email) # Pass session
    # Here you would integrate with an email sending service
    print(f"Email verification token for {request.email}: {token}")
    return {"message": "Verification token generated and (simulated) sent to email."}

@users_router.post("/verify-email/", summary="Verify user email with token")
async def verify_email_with_token(
    verification: EmailVerification,
    session: AsyncSession = Depends(get_db_session) # Inject session
):
    """
    Verifies a user's email using the provided token.
    """
    is_verified = await user_db.verify_user_with_token(session, verification.token) # Pass session
    if not is_verified:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired verification token.")
    return {"message": "Email successfully verified."}

@users_router.get("/me", response_model=UserOut, summary="Get current user's profile")
async def get_user_me(current_user: dict = Depends(get_current_user)):
    """
    Retrieves the profile of the currently authenticated user.
    """
    return UserOut(**current_user)

@users_router.put("/me", response_model=UserOut, summary="Update current user's details")
async def update_user_details(
    user_update: UserUpdate,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session) # Inject session
):
    """
    Updates the details of the currently authenticated user.
    Allows updating first name, last name, and optionally the password.
    """
    user_email = current_user["email"]
    updated_fields = user_update.model_dump(exclude_unset=True)  # Get only fields that are set

    if "new_password" in updated_fields and updated_fields["new_password"]:
        await user_db.update_password(session, user_email, updated_fields.pop("new_password")) # Pass session

    if updated_fields:  # Update other fields if any remain
        await user_db.update_user_details(session, user_email, updated_fields) # Pass session

    # Fetch the updated user to return the latest state
    updated_user = await user_db.get_user_by_email(session, user_email) # Pass session
    if not updated_user:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve updated user data.")
    return UserOut(**updated_user)

@users_router.get("/", response_model=List[UserOut], summary="Get all registered users (Admin only)")
async def list_users(
    session: AsyncSession = Depends(get_db_session) # Inject session
):
    """
    Retrieves a list of all registered users.
    (Note: In a production application, this endpoint should be protected by admin authentication.)
    """
    users_data = await user_db.get_all_users(session) # Pass session
    return [UserOut(**user) for user in users_data]

# --- Constants for calculations ---
RHO = 2670  # kg/m³ for Bouguer correction
EARTH_RADIUS_KM = 6371  # Earth's radius in kilometers for Haversine formula

# --- Dependency to get the DataFrame and ensure it's loaded ---
async def get_dataframe_dependency(
    session: AsyncSession = Depends(get_db_session) # Inject session
) -> pd.DataFrame:
    """
    Dependency function to retrieve the gravity data DataFrame from the database.
    Raises HTTPException if no data is loaded.
    """
    df = await application_db.get_gravity_data(session) # Pass session
    if df.empty:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No gravity data loaded. Please upload data first.")
    return df

# --- Haversine distance function ---
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on Earth using the Haversine formula.
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    return EARTH_RADIUS_KM * c

# --- Gravity Data Endpoints (app_router) ---

@app_router.post("/upload-data/", response_model=UploadResponse, summary="Upload Gravity Data CSV")
async def upload_gravity_data(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db_session) # Inject session
):
    """
    Uploads a CSV file containing gravity data.
    The CSV must have 'latitude', 'longitude', and 'gravity' columns.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file format. Please upload a CSV file.")

    try:
        contents = await file.read()
        row_count = await application_db.load_gravity_data_from_csv(session, contents) # Pass session
        return UploadResponse(message=f"Successfully uploaded {file.filename}", row_count=row_count)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process file: {e}")

@app_router.get("/data/", response_model=List[ProcessedGravityData], summary="Retrieve All Gravity Data")
async def get_all_gravity_data_api(df: pd.DataFrame = Depends(get_dataframe_dependency)):
    """
    Retrieves all loaded gravity data, including any processed fields.
    """
    return df.to_dict(orient="records")

@app_router.post("/clear-data/", summary="Clear All Loaded Gravity Data")
async def clear_all_gravity_data_api(
    session: AsyncSession = Depends(get_db_session) # Inject session
):
    """
    Clears all gravity data currently loaded in the database.
    """
    await application_db.clear_gravity_data(session) # Pass session
    return {"message": "All gravity data cleared from the database."}

@app_router.get("/bouguer-anomaly/", response_model=List[ProcessedGravityData], summary="Calculate Bouguer Anomaly")
async def calculate_bouguer_anomaly(
    df: pd.DataFrame = Depends(get_dataframe_dependency),
    session: AsyncSession = Depends(get_db_session) # Inject session
):
    """
    Calculates the Bouguer anomaly for the loaded gravity data.
    Requires 'elevation' and 'gravity' columns.
    """
    if 'elevation' not in df.columns or 'gravity' not in df.columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing 'elevation' or 'gravity' column in data.")

    # Apply Bouguer correction
    df["bouguer"] = df["gravity"] - (0.3086 * df["elevation"]) + (0.0419 * (RHO / 1000) * df["elevation"])
    await application_db.update_gravity_data(session, df[['id', 'bouguer']]) # Pass session
    return df.to_dict(orient="records")

@app_router.get("/kmeans-clusters/", response_model=List[ClusteringResult], summary="Perform K-Means Clustering")
async def perform_kmeans_clustering(
    n_clusters: int = 3,
    df: pd.DataFrame = Depends(get_dataframe_dependency),
    session: AsyncSession = Depends(get_db_session) # Inject session
):
    """
    Performs K-Means clustering on Latitude, Longitude, Elevation, and Gravity.
    Returns data points with their assigned cluster.
    """
    if n_clusters < 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="n_clusters must be at least 1.")

    features = df[['latitude', 'longitude', 'elevation', 'gravity']]
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(features)
        await application_db.update_gravity_data(session, df[['id', 'cluster']]) # Pass session
        return df[['latitude', 'longitude', 'elevation', 'gravity', 'cluster']].to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"K-Means clustering failed: {e}")

@app_router.get("/anomaly-detection/", response_model=List[AnomalyDetectionResult], summary="Perform Isolation Forest Anomaly Detection")
async def perform_anomaly_detection(
    contamination: float = 0.05,
    df: pd.DataFrame = Depends(get_dataframe_dependency),
    session: AsyncSession = Depends(get_db_session) # Inject session
):
    """
    Performs anomaly detection using Isolation Forest on 'latitude', 'longitude', 'elevation', and 'gravity'.
    Returns data points along with an 'anomaly' flag (-1 for anomaly, 1 for normal).
    """
    if not (0 < contamination < 0.5):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Contamination must be between 0 and 0.5.")

    features = df[['latitude', 'longitude', 'elevation', 'gravity']]
    try:
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        df['anomaly'] = iso_forest.fit_predict(features)
        await application_db.update_gravity_data(session, df[['id', 'anomaly']]) # Pass session
        return df[['latitude', 'longitude', 'elevation', 'gravity', 'anomaly']].to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Isolation Forest anomaly detection failed: {e}")

@app_router.get("/earthquakes/", response_model=List[Dict[str, Any]], summary="Retrieve Earthquake Data")
async def get_earthquakes_api(
    start_date: datetime,
    end_date: datetime,
    min_mag: Optional[float] = None,
    max_mag: Optional[float] = None,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    session: AsyncSession = Depends(get_db_session) # Inject session
):
    """
    Retrieves earthquake data within a specified date range and optional magnitude/depth filters.
    """
    query = EarthquakeQuery(
        start_date=start_date,
        end_date=end_date,
        min_mag=min_mag,
        max_mag=max_mag,
        min_depth=min_depth,
        max_depth=max_depth
    )
    earthquake_data = await application_db.get_earthquakes(session, query) # Pass session
    if not earthquake_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No earthquake data found for the given criteria.")
    return earthquake_data

@app_router.get("/gravity-heatmap/", response_model=PlotlyGraph, summary="Generate Gravity Heatmap")
async def generate_gravity_heatmap(df: pd.DataFrame = Depends(get_dataframe_dependency)):
    """
    Generates an interactive heatmap of gravity anomalies.
    Requires processed data with 'bouguer' anomalies.
    """
    if 'bouguer' not in df.columns or df['bouguer'].isnull().all():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Bouguer anomaly data not available. Please run /bouguer-anomaly/ first.")

    # Create a grid for interpolation
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()

    grid_lat, grid_lon = np.mgrid[lat_min:lat_max:100j, lon_min:lon_max:100j]
    grid_bouguer = griddata(
        (df['latitude'], df['longitude']),
        df['bouguer'],
        (grid_lat, grid_lon),
        method='cubic' # or 'linear', 'nearest'
    )

    fig = go.Figure(data=go.Contour(
        z=grid_bouguer,
        x=grid_lon[0,:],
        y=grid_lat[:,0],
        colorscale='Jet',
        colorbar=dict(title='Bouguer Anomaly (mGal)'),
        line_smoothing=0.85 # Smooth the contours
    ))

    fig.update_layout(
        title='Bouguer Anomaly Heatmap',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        height=600,
        width=800
    )

    return PlotlyGraph(**json.loads(fig.to_json()))

@app_router.get("/cluster-map/", response_model=PlotlyGraph, summary="Generate K-Means Cluster Map")
async def generate_cluster_map(df: pd.DataFrame = Depends(get_dataframe_dependency)):
    """
    Generates an interactive scatter map showing K-Means clusters.
    Requires data processed with K-Means clustering.
    """
    if 'cluster' not in df.columns or df['cluster'].isnull().all():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cluster data not available. Please run /kmeans-clusters/ first.")

    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="cluster",
        size_max=10,
        zoom=3,
        mapbox_style="open-street-map",
        title="K-Means Clustering of Gravity Data",
        hover_name="id"
    )

    fig.update_layout(height=600, width=800)
    return PlotlyGraph(**json.loads(fig.to_json()))

@app_router.get("/anomaly-map/", response_model=PlotlyGraph, summary="Generate Anomaly Detection Map")
async def generate_anomaly_map(df: pd.DataFrame = Depends(get_dataframe_dependency)):
    """
    Generates an interactive scatter map showing anomaly detection results.
    Highlights anomalous data points.
    Requires data processed with anomaly detection.
    """
    if 'anomaly' not in df.columns or df['anomaly'].isnull().all():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Anomaly detection data not available. Please run /anomaly-detection/ first.")

    # Map anomaly values to colors for better visualization
    df['anomaly_color'] = df['anomaly'].map({1: 'blue', -1: 'red'}) # -1 is anomaly, 1 is normal

    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="anomaly_color",
        size_max=10,
        zoom=3,
        mapbox_style="open-street-map",
        title="Gravity Data Anomaly Detection",
        hover_name="id",
        color_discrete_map={'blue': 'Normal', 'red': 'Anomaly'} # Legend mapping
    )

    fig.update_layout(height=600, width=800)
    return PlotlyGraph(**json.loads(fig.to_json()))


# --- Q&A Endpoints (qna_router) ---

@qna_router.post("/questions/", response_model=QuestionResponse, status_code=status.HTTP_201_CREATED, summary="Create a new question")
async def create_new_question(
    question: QuestionCreate,
    session: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user) # Requires authentication
):
    """
    Creates a new question in the Q&A forum.
    The user sending the request is automatically set as the author.
    """
    try:
        new_question_data = await application_db.create_question_db(session, question.text, current_user["id"])
        # Fetch the complete question details including counts and comments
        return await application_db.get_question_by_id_with_details(session, new_question_data.id)
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create question: {e}")

@qna_router.get("/questions/", response_model=List[QuestionResponse], summary="Get all questions with details")
async def get_all_questions(session: AsyncSession = Depends(get_db_session)):
    """
    Retrieves all questions from the Q&A forum,
    including their comments, like counts, and dislike counts.
    """
    try:
        questions_with_details = await application_db.get_all_questions_with_details(session)
        if not questions_with_details:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No questions found.")
        return questions_with_details
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve questions: {e}")

@qna_router.post("/questions/{question_id}/comments/", response_model=CommentResponse, status_code=status.HTTP_201_CREATED, summary="Add a comment to a question")
async def add_comment_to_question(
    question_id: str,
    comment: CommentCreate,
    session: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user) # Requires authentication
):
    """
    Adds a new comment to a specific question.
    """
    # Optional: Verify question_id exists before adding comment
    question_exists = await application_db.get_question_by_id_db(session, question_id)
    if not question_exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found.")

    try:
        new_comment_data = await application_db.create_comment_db(session, question_id, comment.text)
        await session.commit()
        return new_comment_data
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to add comment: {e}")

@qna_router.post("/questions/{question_id}/interact/", response_model=QuestionInteractionResponse, status_code=status.HTTP_201_CREATED, summary="Like or dislike a question")
async def interact_with_question(
    question_id: str,
    interaction_type: LikeDislikeType, # Pydantic Enum for 'like' or 'dislike'
    session: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user) # Requires authentication
):
    """
    Allows a user to like or dislike a question.
    If a user already has an interaction (like/dislike), it updates it.
    """
    user_id = current_user["id"]

    # First, check if the question exists
    question_exists = await application_db.get_question_by_id_db(session, question_id)
    if not question_exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found.")

    # Check for existing interaction by this user on this question
    existing_interaction = await application_db.get_user_question_interaction_db(session, user_id, question_id)

    try:
        if existing_interaction:
            # If exists and type is the same, no change needed. Otherwise, update.
            if existing_interaction.type == interaction_type.value:
                return QuestionInteractionResponse(**existing_interaction._asdict()) # Return existing without change
            else:
                updated_interaction = await application_db.update_question_interaction_db(session, existing_interaction.id, interaction_type)
                await session.commit()
                return updated_interaction
        else:
            # No existing interaction, create a new one
            new_interaction = await application_db.create_question_interaction_db(session, question_id, user_id, interaction_type)
            await session.commit()
            return new_interaction
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to interact with question: {e}")


@qna_router.delete("/questions/{question_id}/interact/{interaction_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a question like/dislike interaction")
async def delete_question_interaction(
    question_id: str, # For validation, though not directly used in deletion query
    interaction_id: str,
    session: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user) # Requires authentication
):
    """
    Deletes a specific like/dislike interaction for a question.
    A user can only delete their own interactions.
    """
    user_id = current_user["id"]

    # First, verify that the interaction exists and belongs to the current user
    interaction = await application_db.get_user_question_interaction_db(session, user_id, question_id)
    if not interaction or interaction.id != interaction_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUDN, detail="Interaction not found or you don't have permission to delete it.")

    try:
        await application_db.delete_question_interaction_db(session, interaction_id)
        await session.commit()
        return {"message": "Interaction deleted successfully"}
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete interaction: {e}")


# --- Researcher Endpoints (researcher_router) ---

@researcher_router.post("/", response_model=Researcher, status_code=status.HTTP_201_CREATED, summary="Create a new researcher entry")
async def create_researcher(
    researcher: Researcher,
    session: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user) # Requires authentication
):
    """
    Creates a new researcher entry (e.g., for a publication).
    """
    try:
        new_researcher = await application_db.create_researcher_db(session, researcher.model_dump())
        if not new_researcher:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create researcher entry.")
        return Researcher(**new_researcher)
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create researcher: {e}")

@researcher_router.get("/", response_model=List[Researcher], summary="Get all researcher entries")
async def get_all_researchers(
    session: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user) # Requires authentication
):
    """
    Retrieves all researcher entries.
    """
    try:
        researchers_data = await application_db.get_all_researchers_db(session)
        return [Researcher(**r) for r in researchers_data]
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve researchers: {e}")

@researcher_router.get("/{researcher_id}", response_model=Researcher, summary="Get a researcher entry by ID")
async def get_researcher_by_id(
    researcher_id: str,
    session: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user) # Requires authentication
):
    """
    Retrieves a single researcher entry by its ID.
    """
    researcher = await application_db.get_researcher_by_id_db(session, researcher_id)
    if not researcher:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Researcher not found.")
    return Researcher(**researcher)

@researcher_router.put("/{researcher_id}", response_model=Researcher, summary="Update a researcher entry")
async def update_researcher(
    researcher_id: str,
    researcher_update: Researcher, # Use the full Researcher schema for updates
    session: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user) # Requires authentication
):
    """
    Updates an existing researcher entry.
    """
    existing_researcher = await application_db.get_researcher_by_id_db(session, researcher_id)
    if not existing_researcher:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Researcher not found.")

    update_data = researcher_update.model_dump(exclude_unset=True) # Get only fields that are set
    if not update_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided for update.")

    try:
        updated_researcher = await application_db.update_researcher_db(session, researcher_id, update_data)
        if not updated_researcher:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update researcher entry.")
        return Researcher(**updated_researcher)
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update researcher: {e}")

@researcher_router.delete("/{researcher_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a researcher entry")
async def delete_researcher(
    researcher_id: str,
    session: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user) # Requires authentication
):
    """
    Deletes a researcher entry by its ID.
    """
    existing_researcher = await application_db.get_researcher_by_id_db(session, researcher_id)
    if not existing_researcher:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Researcher not found.")

    try:
        deleted = await application_db.delete_researcher_db(session, researcher_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete researcher entry.")
        return {"message": "Researcher deleted successfully"}
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete researcher: {e}")

