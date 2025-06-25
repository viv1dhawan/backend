# database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from databases import Database
import sqlalchemy

# Configuration for your PostgreSQL database
DATABASE_URL_ASYNC = "postgresql+asyncpg://geoverse_user:UoeVIoXyhhWruxWADLyKZdcbhEbvD9n1@dpg-d159fj3e5dus739dr010-a.oregon-postgres.render.com/geoverse"
DATABASE_URL_SYNC = "postgresql://geoverse_user:UoeVIoXyhhWruxWADLyKZdcbhEbvD9n1@dpg-d159fj3e5dus739dr010-a.oregon-postgres.render.com/geoverse"

# Async SQLAlchemy engine
async_engine = create_async_engine(DATABASE_URL_ASYNC, echo=False)
AsyncSessionLocal = sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)

# Sync engine for Alembic/migrations
sync_engine = create_engine(DATABASE_URL_SYNC, echo=False)

# Shared metadata
metadata = MetaData()

# Databases library for simple async querying
database = Database(DATABASE_URL_ASYNC)

# Create tables from metadata
async def create_table():
    async with async_engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    print("Tables checked/created.")

# Dependency for FastAPI routes
async def get_db_session():
    async with AsyncSessionLocal() as session:
        yield session