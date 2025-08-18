# database.py
import aiomysql
import asyncio
import os
# from contextlib import asynccontextmanager # REMOVED: No longer needed for get_db_connection

# For metadata.create_all() with models.py tables
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import MetaData
import sqlalchemy # Needed for sqlalchemy.Column, etc., in models.py

# Configuration for your MySQL database
DB_USER = "root"
DB_PASSWORD = "EYDIQLYoErENIIPBMucXrLEHRJmtuEVO"
DB_HOST = "shortline.proxy.rlwy.net"
DB_PORT = 52270
DB_NAME = "railway"

# SQLAlchemy-compatible URL for create_async_engine
DATABASE_URL_ASYNC = f"mysql+aiomysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Global variable for the aiomysql connection pool
db_pool = None

# Shared metadata object for SQLAlchemy table definitions
# This needs to be defined here and imported by models.py
metadata = MetaData()

async def create_db_pool():
    """
    Initializes the aiomysql connection pool.
    This should be called once at application startup.
    """
    global db_pool
    if db_pool is None:
        try:
            db_pool = await aiomysql.create_pool(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASSWORD,
                db=DB_NAME,
                autocommit=False, # We will manage transactions manually
                charset='utf8mb4',
                cursorclass=aiomysql.cursors.DictCursor, # Return results as dictionaries
                minsize=1,
                maxsize=10
            )
            print("Database connection pool created successfully.")
        except Exception as e:
            print(f"Failed to create database pool: {e}")
            raise # Re-raise to prevent app from starting without DB connection

async def close_db_pool():
    """
    Closes the aiomysql connection pool.
    This should be called once at application shutdown.
    """
    global db_pool
    if db_pool:
        db_pool.close()
        await db_pool.wait_closed()
        print("Database connection pool closed.")

async def create_tables():
    """
    Creates database tables based on the SQLAlchemy metadata defined in models.py.
    This function uses a temporary SQLAlchemy AsyncEngine for DDL operations.
    """
    # Import models here to ensure they register with the metadata object
    # This import needs to happen AFTER metadata is defined.
    import models # noqa: F401, E402 (Ignore F401 for unused import, E402 for import position)

    temp_async_engine = create_async_engine(DATABASE_URL_ASYNC, echo=False)
    async with temp_async_engine.begin() as conn:
        # Use run_sync to execute synchronous DDL operations within an async context
        # This will create tables defined in models.py via the shared metadata
        await conn.run_sync(metadata.create_all)
    await temp_async_engine.dispose() # Dispose of the temporary engine
    print("Tables checked/created from models.py definitions.")


# Dependency for FastAPI routes to get an async database connection and cursor
# CHANGED: Removed @asynccontextmanager. This function is now a regular async generator.
# FastAPI's Depends will handle yielding and cleanup through the 'finally' block.
async def get_db_connection():
    """
    Dependency that provides an aiomysql connection and cursor for database interactions.
    The connection is acquired from the pool and released automatically after the request.
    Transactions are managed explicitly within each database operation.
    """
    if db_pool is None:
        await create_db_pool() 

    conn = None
    cursor = None
    try:
        conn = await db_pool.acquire()
        cursor = await conn.cursor()
        yield conn, cursor # This yields the tuple (conn, cursor) to the dependent function
    except Exception as e:
        # If an error occurs, ensure the connection is rolled back if active
        # This is for unhandled exceptions within the API route itself,
        # as db_info functions manage their own transactions.
        if conn:
            await conn.rollback()
        raise e
    finally:
        # This 'finally' block ensures resources are released regardless of success or failure
        if cursor:
            await cursor.close()
        if conn:
            await db_pool.release(conn) # Release the connection back to the pool
