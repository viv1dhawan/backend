from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

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