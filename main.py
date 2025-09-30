import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import create_tables
from app import users_router, app_router, qna_router, researcher_router

app = FastAPI(
    title="Geophysical Data API",
    description="API for managing user accounts, gravity data processing, and earthquake data retrieval.",
    version="1.0.0",
)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(users_router, prefix="/users", tags=["Users"])
app.include_router(app_router, prefix="/app_router", tags=["Geophysical Data"])
app.include_router(qna_router, prefix="/qna_router", tags=["Q&A Forum"])
app.include_router(researcher_router, prefix="/researchers", tags=["Researchers"])

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Geophysical Data API"}

if __name__ == "__main__":
    import uvicorn

    # Railway provides the PORT environment variable
    port = int(os.environ.get("PORT", 8000))

    # Create tables on startup (will not block if DB connection fails)
    try:
        create_tables()
    except Exception as e:
        print(f"Warning: Could not create tables on startup: {e}")

    # Run FastAPI app
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
