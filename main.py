from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Import database and create_table directly, as they are siblings
from database import database, create_table
# Import the APIRouters from app.py, which is a sibling
from app import users_router, app_router
import uvicorn  # Import uvicorn

# Define the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Connects to the database and creates tables on startup,
    and disconnects from the database on shutdown.
    """
    # Startup events
    await database.connect()
    await create_table()
    print("Database connected and tables checked/created.")

    # Print the Swagger UI URL on startup
    # FastAPI's default Swagger UI is at /docs
    swagger_ui_url = "http://127.0.0.1:8000/docs"  # Assuming default host and port
    print(f"Swagger UI available at: {swagger_ui_url}")
    
    yield
    # Shutdown events
    await database.disconnect()
    print("Database disconnected.")

app = FastAPI(
    title="Geophysical Data API",
    description="API for managing user accounts, gravity data processing, and earthquake data retrieval.",
    version="1.0.0",
    lifespan=lifespan  # Assign the lifespan context manager here
)

# Allow CORS for all origins (adjust as necessary for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust origins as needed, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers for different functionalities
app.include_router(users_router, prefix="/users", tags=["Users"])
app.include_router(app_router, prefix="/app_router", tags=["Geophysical Data"])

@app.get("/", summary="Root endpoint")
async def root():
    """
    Root endpoint for the API.
    """
    return {"message": "Welcome to the Geophysical Data API!"}

# This block will only be executed when the script is run directly, not when imported
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)