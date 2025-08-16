from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Import create_table directly, as 'database' object is removed
from database import create_table, async_engine # Import async_engine for connection/disconnection
# Import the APIRouters from app.py, which is a sibling
from app import users_router, app_router, qna_router, researcher_router # Ensure all routers are imported
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
    # The engine is implicitly "connected" when AsyncSessionLocal is used.
    # Explicitly calling async_engine.connect() here can lead to unclosed connections.
    # The create_table() function already handles its own connection management.
    await create_table() # This will ensure tables are created using a managed connection
    print("Database connected and tables checked/created.")

    # Print the Swagger UI URL on startup
    # FastAPI's default Swagger UI is at /docs
    swagger_ui_url = "http://127.0.0.1:8000/docs"  # Assuming default host and port
    print(f"Swagger UI available at: {swagger_ui_url}")
    
    yield
    # Shutdown events
    await async_engine.dispose() # Dispose the engine's connection pool
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
app.include_router(qna_router, prefix="/qna_router", tags=["Q&A Forum "])
app.include_router(researcher_router, prefix="/researchers", tags=["Researchers"]) # Include the new researcher router

@app.get("/", summary="Root endpoint")
async def root():
    """
    Root endpoint for the API, returns a welcome message.
    """
    return {"message": "Welcome to the Geophysical Data API!"}

# This block is typically for running the application directly with `python main.py`
# For production deployment (e.g., with Gunicorn), you would use `uvicorn main:app --host 0.0.0.0 --port 8000`
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
