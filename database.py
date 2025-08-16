# database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from databases import Database
import sqlalchemy
import os

# Use environment variable for security (recommended for production)
# For now, you can hardcode your Railway connection string with sslmode
DATABASE_URL_BASE = "postgresql://your_user:your_password@your_host:your_port/your_db"
DATABASE_URL_ASYNC = f"postgresql+asyncpg://your_user:your_password@your_host:your_port/your_db?sslmode=require"
DATABASE_URL_SYNC = f"{DATABASE_URL_BASE}?sslmode=require"

# Async SQLAlchemy engine
async_engine = create_async_engine(DATABASE_URL_ASYNC, echo=False)
AsyncSessionLocal = sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)

# Sync engine for Alembic/migrations
sync_engine = create_engine(DATABASE_URL_SYNC, echo=False)

# Shared metadata
metadata = MetaData()

# Databases library (good for simple queries)
database = Database(DATABASE_URL_ASYNC)

# Create tables
async def create_table():
    async with async_engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    print("Tables checked/created.")

# Dependency injection for FastAPI
async def get_db_session():
    async with AsyncSessionLocal() as session:
        yield session
