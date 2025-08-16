# database.py

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from databases import Database

# Railway PostgreSQL connection string
DATABASE_URL_ASYNC = "postgresql+asyncpg://postgres:fDoLIsfYMxEhRMQOnnRqpGUhKdKLafKl@yamabiko.proxy.rlwy.net:56650/railway?sslmode=require"
DATABASE_URL_SYNC = "postgresql://postgres:fDoLIsfYMxEhRMQOnnRqpGUhKdKLafKl@yamabiko.proxy.rlwy.net:56650/railway?sslmode=require"

# Async engine for SQLAlchemy
async_engine = create_async_engine(DATABASE_URL_ASYNC, echo=False)
AsyncSessionLocal = sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)

# Sync engine for Alembic migrations
sync_engine = create_engine(DATABASE_URL_SYNC, echo=False)

# Shared metadata (used to define tables/models)
metadata = MetaData()

# Simple async query interface using the `databases` library
database = Database(DATABASE_URL_ASYNC)

# Utility to create all tables
async def create_table():
    async with async_engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    print("Tables created or already exist.")

# Dependency injection for FastAPI
async def get_db_session():
    async with AsyncSessionLocal() as session:
        yield session
