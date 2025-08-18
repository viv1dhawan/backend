# app.py (Extended with Q&A and Researcher Endpoints, updated for JSON POST requests)
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm # Import OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict, Any, Tuple # Import Tuple for type hinting
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
from pydantic import BaseModel
from together import Together # Import Together for AI integration - THIS IS THE ADDED LINE

# Import database operations
import db_info.user_db as user_db
import db_info.application_db as application_db

# Import Pydantic schemas
from Schema.user_schema import UserCreate, UserOut, PasswordReset, PasswordResetRequest, EmailVerificationRequest, EmailVerification, UserUpdate
# Import all new and existing application schemas
from Schema.application_schema import (
    EarthquakeQuery, GravityDataPoint, ProcessedGravityData, TogetherAIRequest, TogetherAIResponse, UploadResponse,
    AnomalyDetectionResult, ClusteringResult, PlotlyGraph, ErrorResponse,
    Researcher, ResearcherDeleteRequest, KMeansRequest, AnomalyDetectionRequest,
    CommentAddRequest, QuestionInteractRequest, QuestionInteractionDeleteRequest,
    ResearcherGetRequest, ResearcherUpdateRequest, QuestionCreate, QuestionResponse,
    CommentResponse, LikeDislikeType, QuestionInteractionResponse
)

# Import database connection dependency
from database import get_db_connection
import aiomysql # Import aiomysql for type hinting connection/cursor

# Define API Routers
users_router = APIRouter()
app_router = APIRouter()
qna_router = APIRouter()
researcher_router = APIRouter()

# --- Removed LoginRequest as OAuth2PasswordRequestForm will be used for /token endpoint ---
# class LoginRequest(BaseModel):
#     username: str
#     password: str

# --- Security Configuration ---
SECRET_KEY = "IAMVIVEKDHAWAN_SUPER_SECRET_KEY"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/token")

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
    # db_context now directly receives the tuple (conn, cursor) from get_db_connection
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection)
) -> Dict[str, Any]:
    """
    Authenticates and retrieves the current user based on the provided JWT token.
    """
    conn, cursor = db_tuple # Unpack the tuple directly
    
    if token in TOKEN_BLACKLIST:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has been revoked")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        current_user = await user_db.get_user_by_email(conn, cursor, email)
        if not current_user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        return current_user
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

# --- User Management Endpoints (users_router) ---

@users_router.post("/signup", response_model=UserOut, summary="Register a new user")
async def signup(
    user: UserCreate,
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection)
):
    """
    Registers a new user with first name, last name, email, and password.
    Hashes the password and sets is_verified to False.
    """
    conn, cursor = db_tuple
    existing_user = await user_db.get_user_by_email(conn, cursor, user.email)
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    created_user_data = await user_db.create_user(conn, cursor, user.model_dump())
    return UserOut(**created_user_data)

@users_router.post("/token", summary="Authenticate user and get access token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection)
):
    """
    Authenticates a user with username (email) and password,
    and returns an access token upon successful login.
    Accepts application/x-www-form-urlencoded body.
    """
    conn, cursor = db_tuple
    current_user = await user_db.get_user_by_email(conn, cursor, form_data.username)
    if not current_user or not user_db.verify_password(form_data.password, current_user["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": current_user["email"]})
    return {"access_token": access_token, "token_type": "bearer"}

@users_router.post("/password-reset-request", summary="Request a password reset token")
async def password_reset_request(
    request: PasswordResetRequest,
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection)
):
    """
    Requests a password reset token for a given email.
    A token will be generated and (simulated) sent to the user's email.
    """
    conn, cursor = db_tuple
    current_user = await user_db.get_user_by_email(conn, cursor, request.email)
    if not current_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    token = await user_db.create_password_reset_token(conn, cursor, current_user["email"])
    print(f"Password reset token for {request.email}: {token}")
    return {"message": "Password reset token generated and (simulated) sent to email.", "token": token}

@users_router.post("/password-reset", summary="Reset user password with token")
async def password_reset(
    password_reset_data: PasswordReset,
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection)
):
    """
    Resets a user's password using the provided token and new password.
    """
    conn, cursor = db_tuple
    email = await user_db.verify_password_reset_token(conn, cursor, password_reset_data.token)
    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired token")
    await user_db.update_password(conn, cursor, email, password_reset_data.new_password)
    return {"message": "Password updated successfully"}

@users_router.post("/request-email-verification/", summary="Request email verification token")
async def request_email_verification(
    request: EmailVerificationRequest,
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection)
):
    """
    Requests an email verification token for a given email.
    In a real app, this would send an email with the token.
    """
    conn, cursor = db_tuple
    current_user = await user_db.get_user_by_email(conn, cursor, request.email)
    if not current_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if current_user["is_verified"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already verified")

    token = await user_db.generate_verification_token(conn, cursor, request.email)
    print(f"Email verification token for {request.email}: {token}")
    return {"message": "Verification token generated and (simulated) sent to email."}

@users_router.post("/verify-email/", summary="Verify user email with token")
async def verify_email_with_token(
    verification: EmailVerification,
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection)
):
    """
    Verifies a user's email using the provided token.
    """
    conn, cursor = db_tuple
    is_verified = await user_db.verify_user_with_token(conn, cursor, verification.token)
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
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection)
):
    """
    Updates the details of the currently authenticated user.
    Allows updating first name, last name, and optionally the password.
    """
    conn, cursor = db_tuple
    user_email = current_user["email"]
    updated_fields = user_update.model_dump(exclude_unset=True)

    if "new_password" in updated_fields and updated_fields["new_password"]:
        await user_db.update_password(conn, cursor, user_email, updated_fields.pop("new_password"))

    if updated_fields:
        await user_db.update_user_details(conn, cursor, user_email, updated_fields)

    updated_user = await user_db.get_user_by_email(conn, cursor, user_email)
    if not updated_user:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve updated user data.")
    return UserOut(**updated_user)

@users_router.get("/", response_model=List[UserOut], summary="Get all registered users (Admin only)")
async def list_users(
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection)
):
    """
    Retrieves a list of all registered users.
    (Note: In a production application, this endpoint should be protected by admin authentication.)
    """
    conn, cursor = db_tuple
    users_data = await user_db.get_all_users(conn, cursor)
    return [UserOut(**user) for user in users_data]

# --- Constants for calculations ---
RHO = 2670  # kg/m³ for Bouguer correction
EARTH_RADIUS_KM = 6371  # Earth's radius in kilometers for Haversine formula

# --- Dependency to get the DataFrame and ensure it's loaded ---
async def get_dataframe_dependency(
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection)
) -> pd.DataFrame:
    """
    Dependency function to retrieve the gravity data DataFrame from the database.
    Raises HTTPException if no data is loaded.
    """
    conn, cursor = db_tuple
    df = await application_db.get_gravity_data(conn, cursor)
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
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection)
):
    """
    Uploads a CSV file containing gravity data.
    The CSV must have 'latitude', 'longitude', and 'gravity' columns.
    This endpoint expects 'multipart/form-data' for file upload.
    """
    conn, cursor = db_tuple
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file format. Please upload a CSV file.")

    try:
        contents = await file.read()
        row_count = await application_db.load_gravity_data_from_csv(conn, cursor, contents)
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
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection)
):
    """
    Clears all gravity data currently loaded in the database.
    """
    conn, cursor = db_tuple
    await application_db.clear_gravity_data(conn, cursor)
    return {"message": "All gravity data cleared from the database."}

# Changed from GET to POST and uses KMeansRequest for n_clusters
@app_router.post("/kmeans-clusters/", response_model=List[ClusteringResult], summary="Perform K-Means Clustering (via JSON body)")
async def perform_kmeans_clustering(
    request: KMeansRequest,
    df: pd.DataFrame = Depends(get_dataframe_dependency),
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection)
):
    """
    Performs K-Means clustering on Latitude, Longitude, Elevation, and Gravity.
    Returns data points with their assigned cluster.
    Expects JSON body: {"n_clusters": 3}
    """
    conn, cursor = db_tuple
    n_clusters = request.n_clusters
    if n_clusters < 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="n_clusters must be at least 1.")

    features = df[['latitude', 'longitude', 'elevation', 'gravity']]
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(features)
        await application_db.update_gravity_data(conn, cursor, df[['id', 'cluster']])
        return df[['latitude', 'longitude', 'elevation', 'gravity', 'cluster']].to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"K-Means clustering failed: {e}")

# Changed from GET to POST and uses AnomalyDetectionRequest for contamination
@app_router.post("/anomaly-detection/", response_model=List[AnomalyDetectionResult], summary="Perform Isolation Forest Anomaly Detection (via JSON body)")
async def perform_anomaly_detection(
    request: AnomalyDetectionRequest,
    df: pd.DataFrame = Depends(get_dataframe_dependency),
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection)
):
    """
    Performs anomaly detection using Isolation Forest on 'latitude', 'longitude', 'elevation', and 'gravity'.
    Returns data points along with an 'anomaly' flag (-1 for anomaly, 1 for normal).
    Expects JSON body: {"contamination": 0.05}
    """
    conn, cursor = db_tuple
    contamination = request.contamination
    if not (0 < contamination < 0.5):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Contamination must be between 0 and 0.5.")

    features = df[['latitude', 'longitude', 'elevation', 'gravity']]
    try:
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        df['anomaly'] = iso_forest.fit_predict(features)
        await application_db.update_gravity_data(conn, cursor, df[['id', 'anomaly']])
        return df[['latitude', 'longitude', 'elevation', 'gravity', 'anomaly']].to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Isolation Forest anomaly detection failed: {e}")

# Changed from GET to POST and uses EarthquakeQuery for all parameters
@app_router.post("/earthquakes-query/", response_model=List[Dict[str, Any]], summary="Retrieve Earthquake Data (via JSON body)")
async def get_earthquakes_api(
    query_params: EarthquakeQuery, # Now accepts EarthquakeQuery as JSON body
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection)
):
    """
    Retrieves earthquake data within a specified date range and optional magnitude/depth filters.
    Expects JSON body with fields like:
    {
        "start_date": "2023-01-01T00:00:00",
        "end_date": "2023-01-31T23:59:59",
        "min_mag": 2.5,
        "max_mag": 5.0
    }
    """
    conn, cursor = db_tuple
    # Convert Pydantic model to a dictionary suitable for db function
    params_dict = query_params.model_dump(exclude_unset=True)
    earthquake_data = await application_db.get_earthquakes(conn, cursor, params_dict)
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

    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()

    grid_lat, grid_lon = np.mgrid[lat_min:lat_max:100j, lon_min:lon_max:100j]
    grid_bouguer = griddata(
        (df['latitude'], df['longitude']),
        df['bouguer'],
        (grid_lat, grid_lon),
        method='cubic'
    )

    fig = go.Figure(data=go.Contour(
        z=grid_bouguer,
        x=grid_lon[0,:],
        y=grid_lat[:,0],
        colorscale='Jet',
        colorbar=dict(title='Bouguer Anomaly (mGal)'),
        line_smoothing=0.85
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

    df['anomaly_color'] = df['anomaly'].map({1: 'blue', -1: 'red'})

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
        color_discrete_map={'blue': 'Normal', 'red': 'Anomaly'}
    )

    fig.update_layout(height=600, width=800)
    return PlotlyGraph(**json.loads(fig.to_json()))

@app_router.post("/generate-ai-response/", response_model=TogetherAIResponse, summary="Generate AI response using Together AI")
async def generate_ai_response(
    request: TogetherAIRequest,
    current_user: dict = Depends(get_current_user) # Requires authentication
):
    """
    Generates a response from Together AI based on the provided prompt.
    """
    try:
        client = Together(api_key="ca4c86eee93c0e84892aa54f9d17762bf1c5b71e2673dfdc4e81168a9d961125") # Ensure API key is set as environment variable
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": request.prompt
                }
            ]
        )
        generated_text = response.choices[0].message.content
        return TogetherAIResponse(generated_text=generated_text)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate AI response: {e}")


# --- Q&A Endpoints (qna_router) ---

@qna_router.post("/questions/", response_model=QuestionResponse, status_code=status.HTTP_201_CREATED, summary="Create a new question")
async def create_new_question(
    question: QuestionCreate,
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection),
    current_user: dict = Depends(get_current_user)
):
    """
    Creates a new question in the Q&A forum.
    The user sending the request is automatically set as the author.
    """
    conn, cursor = db_tuple
    try:
        new_question_data = await application_db.create_question_db(conn, cursor, question.text, current_user["id"])
        return QuestionResponse(**new_question_data)
    except Exception as e:
        await conn.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create question: {e}")

@qna_router.get("/questions/", response_model=List[QuestionResponse], summary="Get all questions with details")
async def get_all_questions(db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection)):
    """
    Retrieves all questions from the Q&A forum,
    including their comments, like counts, and dislike counts.
    """
    conn, cursor = db_tuple
    try:
        questions_with_details = await application_db.get_all_questions_with_details(conn, cursor)
        if not questions_with_details:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No questions found.")
        return [QuestionResponse(**q) for q in questions_with_details]
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve questions: {e}")

# Changed from path parameter to JSON body for question_id and text
@qna_router.post("/comments/", response_model=CommentResponse, status_code=status.HTTP_201_CREATED, summary="Add a comment to a question (via JSON body)")
async def add_comment_to_question(
    request: CommentAddRequest, # Expects JSON body with question_id and text
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection),
    current_user: dict = Depends(get_current_user)
):
    """
    Adds a new comment to a specific question.
    Expects JSON body: {"question_id": "...", "text": "..."}
    """
    conn, cursor = db_tuple
    question_id = request.question_id
    comment_text = request.text

    question_exists = await application_db.get_question_by_id_db(conn, cursor, question_id)
    if not question_exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found.")

    try:
        new_comment_data = await application_db.create_comment_db(conn, cursor, question_id, comment_text)
        await conn.commit()
        return CommentResponse(**new_comment_data)
    except Exception as e:
        await conn.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to add comment: {e}")

# Changed from path and query parameters to JSON body for question_id and interaction_type
@qna_router.post("/interact/", response_model=QuestionInteractionResponse, status_code=status.HTTP_201_CREATED, summary="Like or dislike a question (via JSON body)")
async def interact_with_question(
    request: QuestionInteractRequest, # Expects JSON body with question_id and interaction_type
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection),
    current_user: dict = Depends(get_current_user)
):
    """
    Allows a user to like or dislike a question.
    If a user already has an interaction (like/dislike), it updates it.
    Expects JSON body: {"question_id": "...", "interaction_type": "like"}
    """
    conn, cursor = db_tuple
    user_id = current_user["id"]
    question_id = request.question_id
    interaction_type_value = request.interaction_type.value # Access the string value of the enum

    question_exists = await application_db.get_question_by_id_db(conn, cursor, question_id)
    if not question_exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found.")

    existing_interaction = await application_db.get_user_question_interaction_db(conn, cursor, user_id, question_id)

    try:
        if existing_interaction:
            if existing_interaction['type'] == interaction_type_value:
                # If the existing interaction is the same type, return it without creating a new one
                return QuestionInteractionResponse(**existing_interaction)
            else:
                # If the existing interaction is different, update it
                updated_interaction = await application_db.update_question_interaction_db(conn, cursor, existing_interaction['id'], interaction_type_value)
                await conn.commit()
                return QuestionInteractionResponse(**updated_interaction)
        else:
            # No existing interaction, create a new one
            new_interaction = await application_db.create_question_interaction_db(conn, cursor, question_id, user_id, interaction_type_value)
            await conn.commit()
            return QuestionInteractionResponse(**new_interaction)
    except Exception as e:
        await conn.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to interact with question: {e}")

# Changed from path parameters to JSON body for question_id and interaction_id
@qna_router.delete("/interaction-delete/", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a question like/dislike interaction (via JSON body)")
async def delete_question_interaction(
    request: QuestionInteractionDeleteRequest, # Expects JSON body with question_id and interaction_id
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection),
    current_user: dict = Depends(get_current_user)
):
    """
    Deletes a specific like/dislike interaction for a question.
    A user can only delete their own interactions.
    Expects JSON body: {"question_id": "...", "interaction_id": "..."}
    """
    conn, cursor = db_tuple
    user_id = current_user["id"]
    question_id = request.question_id
    interaction_id_to_delete = request.interaction_id

    # Verify that the interaction exists and belongs to the current user and question
    interaction = await application_db.get_user_question_interaction_db(conn, cursor, user_id, question_id)
    if not interaction or str(interaction['id']) != interaction_id_to_delete: # Convert interaction['id'] to string for comparison
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interaction not found or you don't have permission to delete it.")

    try:
        await application_db.delete_question_interaction_db(conn, cursor, interaction_id_to_delete)
        await conn.commit()
        return {"message": "Interaction deleted successfully"}
    except Exception as e:
        await conn.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete interaction: {e}")


# --- Researcher Endpoints (researcher_router) ---

@researcher_router.post("/add_researcher", response_model=Researcher, status_code=status.HTTP_201_CREATED, summary="Create a new researcher entry")
async def create_researcher(
    researcher: Researcher,
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection),
    current_user: dict = Depends(get_current_user)
):
    """
    Creates a new researcher entry (e.g., for a publication).
    """
    conn, cursor = db_tuple
    try:
        new_researcher = await application_db.create_researcher_db(conn, cursor, researcher.model_dump())
        if not new_researcher:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create researcher entry.")
        return Researcher(**new_researcher)
    except Exception as e:
        await conn.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create researcher: {e}")

@researcher_router.get("/", response_model=List[Researcher], summary="Get all researcher entries")
async def get_all_researchers(
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection),
    current_user: dict = Depends(get_current_user)
):
    """
    Retrieves all researcher entries.
    """
    conn, cursor = db_tuple
    try:
        researchers_data = await application_db.get_all_researchers_db(conn, cursor)
        return [Researcher(**r) for r in researchers_data]
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve researchers: {e}")

# Changed from GET with path parameter to POST with JSON body
@researcher_router.post("/get-by-id/", response_model=Researcher, summary="Get a researcher entry by ID (via JSON body)")
async def get_researcher_by_id(
    request: ResearcherGetRequest, # Expects JSON body with researcher_id
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection),
    current_user: dict = Depends(get_current_user)
):
    """
    Retrieves a single researcher entry by its ID, provided in the request body as JSON.
    Expects JSON body: {"researcher_id": "123"}
    """
    conn, cursor = db_tuple
    researcher_id = request.researcher_id
    researcher = await application_db.get_researcher_by_id_db(conn, cursor, researcher_id)
    if not researcher:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Researcher not found.")
    return Researcher(**researcher)

# Changed from PUT with path parameter and body to POST with JSON body containing ID
@researcher_router.post("/update-researcher/", response_model=Researcher, summary="Update a researcher entry (via JSON body)")
async def update_researcher(
    request: ResearcherUpdateRequest, # Expects JSON body with researcher_id and update fields
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection),
    current_user: dict = Depends(get_current_user)
):
    """
    Updates an existing researcher entry.
    Expects JSON body with `researcher_id` and fields to update, e.g.:
    {"researcher_id": "123", "title": "New Title", "authors": "Updated Authors"}
    """
    conn, cursor = db_tuple
    researcher_id = request.researcher_id
    
    # Extract update_data from the request model, excluding researcher_id and unset fields
    update_data = request.model_dump(exclude_unset=True)
    update_data.pop("researcher_id", None) # Ensure researcher_id is not in update_data for the DB function

    if not update_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided for update.")

    existing_researcher = await application_db.get_researcher_by_id_db(conn, cursor, researcher_id)
    if not existing_researcher:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Researcher not found.")

    try:
        updated_researcher = await application_db.update_researcher_db(conn, cursor, researcher_id, update_data)
        if not updated_researcher:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update researcher entry.")
        return Researcher(**updated_researcher)
    except Exception as e:
        await conn.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update researcher: {e}")

# This endpoint was already updated in the previous turn
@researcher_router.delete("/delete-researcher/", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a researcher entry by ID (via JSON body)")
async def delete_researcher(
    request: ResearcherDeleteRequest, # Expects a JSON body with researcher_id
    db_tuple: Tuple[aiomysql.Connection, aiomysql.DictCursor] = Depends(get_db_connection),
    current_user: dict = Depends(get_current_user) # Requires authentication
):
    """
    Deletes a researcher entry by its ID, provided in the request body as JSON.
    Example JSON Body: {"researcher_id": "123"}
    """
    conn, cursor = db_tuple
    researcher_id_to_delete = request.researcher_id # Extract ID from the JSON body

    existing_researcher = await application_db.get_researcher_by_id_db(conn, cursor, researcher_id_to_delete)
    if not existing_researcher:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Researcher not found.")

    try:
        deleted = await application_db.delete_researcher_db(conn, cursor, researcher_id_to_delete)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete researcher entry.")
        return {"message": "Researcher deleted successfully"}
    except Exception as e:
        await conn.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete researcher: {e}")

