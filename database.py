# database.py

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import MetaData
from sqlalchemy.orm import sessionmaker
from databases import Database
import sqlalchemy

# Use your actual Railway DB URL with sslmode=require
DATABASE_URL = "postgresql://postgres:fDoLIsfYMxEhRMQOnnRqpGUhKdKLafKl@yamabiko.proxy.rlwy.net:56650/railway"

# Async URLs for asyncpg + sqlalchemy
DATABASE_URL_ASYNC = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://") + "?sslmode=require"

# Async SQLAlchemy engine & session
async_engine = create_async_engine(DATABASE_URL_ASYNC, echo=False)
AsyncSessionLocal = sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)

# Shared metadata (should be used in your models)
metadata = MetaData()

# Optional: `databases` library for async queries
database = Database(DATABASE_URL_ASYNC)

#  Create tables (run once at startup)
async def create_table():
    #  Ensure all your models are imported before this line
    import models  # Adjust based on your actual models file
    async with async_engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    print(" Tables checked/created.")

#  FastAPI dependency
async def get_db_session():
    async with AsyncSessionLocal() as session:
        yield session
