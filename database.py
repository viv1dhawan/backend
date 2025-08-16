# database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
import sqlalchemy

# Configuration for your MySQL database
# Use the connection string provided by the user
DATABASE_URL_ASYNC = "mysql+aiomysql://root:EYDIQLYoErENIIPBMucXrLEHRJmtuEVO@shortline.proxy.rlwy.net:52270/railway"
DATABASE_URL_SYNC = "mysql+mysqlconnector://root:EYDIQLYoErENIIPBMucXrLEHRJmtuEVO@shortline.proxy.rlwy.net:52270/railway"

# Async SQLAlchemy engine
# 'aiomysql' is used for async, as 'mysql-connector-python-async' needs specific dialect
# For sync, 'mysqlconnector' is a common choice
async_engine = create_async_engine(DATABASE_URL_ASYNC, echo=False)
AsyncSessionLocal = sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)

# Sync engine for Alembic/migrations or simple sync operations (if needed)
sync_engine = create_engine(DATABASE_URL_SYNC, echo=False)

# Shared metadata
metadata = MetaData()

# Create tables from metadata
async def create_table():
    """
    Creates database tables based on the defined metadata.
    This function should be called during application startup.
    """
    async with async_engine.begin() as conn:
        # Use run_sync to execute synchronous DDL operations within an async context
        await conn.run_sync(metadata.create_all)
    print("Tables checked/created.")

# Dependency for FastAPI routes to get an async database session
async def get_db_session():
    """
    Dependency that provides an AsyncSession for database interactions.
    The session is automatically closed after the request.
    """
    async with AsyncSessionLocal() as session:
        yield session
