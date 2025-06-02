from fastapi import APIRouter, HTTPException, Depends, Form, File, UploadFile, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import List, Optional
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

# Import database operations from direct sub-packages
import db_info.user_db as user_db
import db_info.application_db as application_db

# Import Pydantic schemas from direct sub-packages
from Schema.user_schema import UserCreate, UserOut, PasswordReset, PasswordResetRequest, EmailVerificationRequest, EmailVerification, UserUpdate
from Schema.gravity_schema import EarthquakeQuery, GravityDataPoint, ProcessedGravityData, UploadResponse, AnomalyDetectionResult, ClusteringResult, PlotlyGraph, ErrorResponse

# Define API Routers
users_router = APIRouter()
app_router = APIRouter()

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

async def get_current_user(token: str = Depends(oauth2_scheme)):
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
        current_user = await user_db.get_user_by_email(email)
        if not current_user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        return current_user
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

# --- User Management Endpoints (users_router) ---

@users_router.post("/signup", response_model=UserOut, summary="Register a new user")
async def signup(user: UserCreate):
    """
    Registers a new user with first name, last name, email, and password.
    Hashes the password and sets is_verified to False.
    """
    existing_user = await user_db.get_user_by_email(user.email)
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    
    created_user_data = await user_db.create_user(user.model_dump())
    return UserOut(**created_user_data)

@users_router.post("/token", summary="Authenticate user and get access token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticates a user with username (email) and password,
    and returns an access token upon successful login.
    """
    current_user = await user_db.get_user_by_email(form_data.username)
    if not current_user or not user_db.verify_password(form_data.password, current_user["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": current_user["email"]})
    return {"access_token": access_token, "token_type": "bearer"}

@users_router.post("/password-reset-request", summary="Request a password reset token")
async def password_reset_request(request: PasswordResetRequest):
    """
    Requests a password reset token for a given email.
    A token will be generated and (simulated) sent to the user's email.
    """
    current_user = await user_db.get_user_by_email(request.email)
    if not current_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    token = await user_db.create_password_reset_token(current_user["email"])
    # In a real application, you would send this token via email
    print(f"Password reset token for {request.email}: {token}")  # For testing/development
    return {"message": "Password reset token generated and (simulated) sent to email.", "token": token}

@users_router.post("/password-reset", summary="Reset user password with token")
async def password_reset(password_reset_data: PasswordReset):
    """
    Resets a user's password using the provided token and new password.
    """
    email = await user_db.verify_password_reset_token(password_reset_data.token)
    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired token")
    await user_db.update_password(email, password_reset_data.new_password)
    return {"message": "Password updated successfully"}

@users_router.post("/request-email-verification/", summary="Request email verification token")
async def request_email_verification(request: EmailVerificationRequest):
    """
    Requests an email verification token for a given email.
    In a real app, this would send an email with the token.
    """
    current_user = await user_db.get_user_by_email(request.email)
    if not current_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if current_user["is_verified"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already verified")

    token = await user_db.generate_verification_token(request.email)
    # Here you would integrate with an email sending service
    print(f"Email verification token for {request.email}: {token}")
    return {"message": "Verification token generated and (simulated) sent to email."}

@users_router.post("/verify-email/", summary="Verify user email with token")
async def verify_email_with_token(verification: EmailVerification):
    """
    Verifies a user's email using the provided token.
    """
    is_verified = await user_db.verify_user_with_token(verification.token)
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
    current_user: dict = Depends(get_current_user)
):
    """
    Updates the details of the currently authenticated user.
    Allows updating first name, last name, and optionally the password.
    """
    user_email = current_user["email"]
    updated_fields = user_update.model_dump(exclude_unset=True)  # Get only fields that are set

    if "new_password" in updated_fields and updated_fields["new_password"]:
        await user_db.update_password(user_email, updated_fields.pop("new_password"))
    
    if updated_fields:  # Update other fields if any remain
        await user_db.update_user_details(user_email, updated_fields)

    # Fetch the updated user to return the latest state
    updated_user = await user_db.get_user_by_email(user_email)
    if not updated_user:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve updated user data.")
    return UserOut(**updated_user)

@users_router.get("/", response_model=List[UserOut], summary="Get all registered users (Admin only)")
async def list_users():
    """
    Retrieves a list of all registered users.
    (Note: In a production application, this endpoint should be protected by admin authentication.)
    """
    users_data = await user_db.get_all_users()
    return [UserOut(**user) for user in users_data]

# --- Constants for calculations ---
RHO = 2670  # kg/m³ for Bouguer correction
EARTH_RADIUS_KM = 6371  # Earth's radius in kilometers for Haversine formula

# --- Dependency to get the DataFrame and ensure it's loaded ---
async def get_dataframe_dependency() -> pd.DataFrame:
    """
    Dependency function to retrieve the gravity data DataFrame from the database.
    Raises HTTPException if no data is loaded.
    """
    df = await application_db.get_gravity_data()
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
async def upload_gravity_data(file: UploadFile = File(...)):
    """
    Uploads a CSV file containing gravity data.
    The CSV must have 'latitude', 'longitude', 'elevation', and 'gravity' columns.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file format. Please upload a CSV file.")

    try:
        contents = await file.read()
        row_count = await application_db.load_gravity_data_from_csv(contents)
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
async def clear_all_gravity_data_api():
    """
    Clears all gravity data currently loaded in the database.
    """
    await application_db.clear_gravity_data()
    return {"message": "All gravity data cleared from the database."}

@app_router.get("/bouguer-anomaly/", response_model=List[ProcessedGravityData], summary="Calculate Bouguer Anomaly")
async def calculate_bouguer_anomaly(df: pd.DataFrame = Depends(get_dataframe_dependency)):
    """
    Calculates the Bouguer anomaly for the loaded gravity data.
    Requires 'elevation' and 'gravity' columns.
    """
    if 'elevation' not in df.columns or 'gravity' not in df.columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing 'elevation' or 'gravity' column in data.")

    # Apply Bouguer correction
    df["bouguer"] = df["gravity"] - (0.3086 * df["elevation"]) + (0.0419 * (RHO / 1000) * df["elevation"])
    await application_db.update_gravity_data(df[['id', 'bouguer']])  # Update only the bouguer column
    return df.to_dict(orient="records")

@app_router.get("/kmeans-clusters/", response_model=List[ClusteringResult], summary="Perform K-Means Clustering")
async def perform_kmeans_clustering(
    n_clusters: int = 3,
    df: pd.DataFrame = Depends(get_dataframe_dependency)
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
        await application_db.update_gravity_data(df[['id', 'cluster']])  # Update only the cluster column
        return df[['latitude', 'longitude', 'elevation', 'gravity', 'cluster']].to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"K-Means clustering failed: {e}")

@app_router.get("/anomaly-detection/", response_model=List[AnomalyDetectionResult], summary="Perform Isolation Forest Anomaly Detection")
async def perform_anomaly_detection(
    contamination: float = 0.05,
    df: pd.DataFrame = Depends(get_dataframe_dependency)
):
    """
    Performs Isolation Forest anomaly detection on Latitude, Longitude, Elevation, and Gravity.
    Returns data points with their anomaly status (-1 for anomaly, 1 for normal).
    """
    if not (0 < contamination < 0.5):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Contamination must be between 0 and 0.5.")

    features = df[['latitude', 'longitude', 'elevation', 'gravity']]
    try:
        iso = IsolationForest(contamination=contamination, random_state=42)
        df["anomaly"] = iso.fit_predict(features)
        await application_db.update_gravity_data(df[['id', 'anomaly']])  # Update only the anomaly column
        return df[['latitude', 'longitude', 'elevation', 'gravity', 'anomaly']].to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Isolation Forest anomaly detection failed: {e}")

@app_router.get("/plot-map-bouguer/", response_model=PlotlyGraph, summary="Generate Bouguer Anomaly Map")
async def plot_map_bouguer(df: pd.DataFrame = Depends(get_dataframe_dependency)):
    """
    Generates a Plotly scatter map visualizing the Bouguer anomaly.
    Requires 'latitude', 'longitude', and 'bouguer' columns.
    """
    if 'bouguer' not in df.columns or df['bouguer'].isnull().all():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Bouguer anomaly not calculated or all values are null. Please run /gravity/bouguer-anomaly first.")

    fig = px.scatter_map(
        df,
        lat="latitude",
        lon="longitude",
        color="bouguer",
        title="Bouguer Anomaly Map",
        color_continuous_scale=px.colors.sequential.Viridis,
        hover_name="bouguer",
        hover_data={"latitude": True, "longitude": True, "elevation": True, "gravity": True, "bouguer": True},
        map_style="open-street-map",
        zoom=6
    )
    # Return Plotly graph as JSON
    return PlotlyGraph(**json.loads(fig.to_json()))

@app_router.get("/plot-map-anomaly/", response_model=PlotlyGraph, summary="Generate Anomaly Detection Map")
async def plot_map_anomaly(df: pd.DataFrame = Depends(get_dataframe_dependency)):
    """
    Generates a Plotly scatter map visualizing the anomaly detection results.
    Requires 'latitude', 'longitude', and 'anomaly' columns.
    """
    if 'anomaly' not in df.columns or df['anomaly'].isnull().all():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Anomaly detection not performed or all values are null. Please run /gravity/anomaly-detection first.")

    fig = px.scatter_map(
        df,
        lat="latitude",
        lon="longitude",
        color="anomaly",
        color_discrete_map={-1: "red", 1: "blue"},  # -1 = anomaly, 1 = normal
        title="Gravity Anomaly Detection",
        hover_name="anomaly",
        hover_data={"latitude": True, "longitude": True, "elevation": True, "gravity": True, "anomaly": True},
        map_style="open-street-map",
        zoom=6
    )
    # Return Plotly graph as JSON
    return PlotlyGraph(**json.loads(fig.to_json()))

@app_router.get("/plot-map-clusters/", response_model=PlotlyGraph, summary="Generate K-Means Clustering Map")
async def plot_map_clusters(df: pd.DataFrame = Depends(get_dataframe_dependency)):
    """
    Generates a Plotly scatter map visualizing the K-Means clustering results.
    Requires 'latitude', 'longitude', and 'cluster' columns.
    """
    if 'cluster' not in df.columns or df['cluster'].isnull().all():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="K-Means clustering not performed or all values are null. Please run /gravity/kmeans-clusters first.")

    fig = px.scatter_map(
        df,
        lat="latitude",
        lon="longitude",
        color="cluster",
        title="Gravity Data K-Means Clusters",
        hover_name="cluster",
        hover_data={"latitude": True, "longitude": True, "elevation": True, "gravity": True, "cluster": True},
        map_style="open-street-map",
        zoom=6
    )
    # Return Plotly graph as JSON
    return PlotlyGraph(**json.loads(fig.to_json()))

@app_router.get("/interpolate-gravity/", response_model=PlotlyGraph, summary="Generate Interpolated Gravity Map")
async def interpolate_gravity(
    grid_resolution: int = 100,
    df: pd.DataFrame = Depends(get_dataframe_dependency)
):
    """
    Interpolates gravity data and generates a contour map.
    Requires 'latitude', 'longitude', and 'gravity' columns.
    """
    if 'gravity' not in df.columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing 'gravity' column in data.")

    try:
        points = df[['latitude', 'longitude']].values
        values = df['gravity'].values

        # Create a grid for interpolation
        lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
        lon_min, lon_max = df['longitude'].min(), df['longitude'].max()

        grid_lat, grid_lon = np.mgrid[lat_min:lat_max:complex(grid_resolution),
                                        lon_min:lon_max:complex(grid_resolution)]

        # Interpolate the gravity values
        grid_gravity = griddata(points, values, (grid_lat, grid_lon), method='cubic')

        # Create the contour map
        fig = go.Figure(data=go.Contour(
            z=grid_gravity,
            x=grid_lon[0, :],
            y=grid_lat[:, 0],
            colorscale='Viridis',
            colorbar_title='Gravity (mGal)'
        ))

        fig.update_layout(
            title='Interpolated Gravity Map',
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            geo_scope='world'  # Ensure the map is displayed globally
        )
        # Return Plotly graph as JSON
        return PlotlyGraph(**json.loads(fig.to_json()))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Gravity interpolation failed: {e}")

@app_router.get("/distance-from-point/", response_model=List[ProcessedGravityData], summary="Calculate Distance from a Reference Point")
async def calculate_distance_from_point(
    ref_lat: float,
    ref_lon: float,
    df: pd.DataFrame = Depends(get_dataframe_dependency)
):
    """
    Calculates the Haversine distance of each data point from a specified reference latitude and longitude.
    """
    df['distance_km'] = df.apply(
        lambda row: haversine(ref_lat, ref_lon, row['latitude'], row['longitude']),
        axis=1
    )
    await application_db.update_gravity_data(df[['id', 'distance_km']])  # Update only the distance_km column
    return df.to_dict(orient="records")

@app_router.post("/earthquakes", summary="Fetch Earthquake Data")
async def fetch_earthquakes(query: EarthquakeQuery):
    """
    Retrieves earthquake data based on specified filters.
    """
    rows = await application_db.get_earthquakes(query)  # Pass the 'query' object
    return [
        {
            "id": row["id"],
            "time": row["time"],
            "mag": row["mag"],
            "depth": row["depth"],
            "place": row["place"],
            "latitude": row["latitude"],
            "longitude": row["longitude"]
        }
        for row in rows
    ]