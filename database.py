# database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from databases import Database
import sqlalchemy

# Configuration for your PostgreSQL database
DATABASE_URL_ASYNC = "postgresql+asyncpg://postgres:Vivek12@localhost:5433/myapp_database"
DATABASE_URL_SYNC = "postgresql://postgres:Vivek12@localhost:5433/myapp_database"

# Async engine for use with FastAPI and async operations
# Change echo=True to echo=False to suppress SQLAlchemy engine logs
async_engine = create_async_engine(DATABASE_URL_ASYNC, echo=False)
AsyncSessionLocal = sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)

# Sync engine for reflection-based table loading (if needed, e.g., for Alembic migrations)
# Change echo=True to echo=False to suppress SQLAlchemy engine logs
sync_engine = create_engine(DATABASE_URL_SYNC, echo=False)

# Metadata object to store table definitions
metadata = MetaData()

# Database object from 'databases' library for simpler async queries
database = Database(DATABASE_URL_ASYNC)

async def create_table():
    """
    Creates all tables defined in models.py if they do not already exist.
    This function should be called during application startup.
    """
    async with async_engine.begin() as conn:
        # Import models here to ensure they are registered with metadata
        # models is now a sibling file
        import models
        await conn.run_sync(metadata.create_all)
    print("Database tables created/checked.")