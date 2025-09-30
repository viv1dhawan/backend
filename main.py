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


# Run with uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn
    # This calls the synchronous function to create tables on startup
    # For a real application, you'd want to handle this more robustly
    create_tables() 
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
