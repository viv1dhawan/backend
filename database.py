# database.py
import aiomysql
import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import MetaData

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
                autocommit=False,
                charset='utf8mb4',
                cursorclass=aiomysql.cursors.DictCursor,
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
    import models 
    temp_async_engine = create_async_engine(DATABASE_URL_ASYNC, echo=False)
    async with temp_async_engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    await temp_async_engine.dispose() # Dispose of the temporary engine
    print("Tables checked/created from models.py definitions.")

async def get_db_connection():
    if db_pool is None:
        await create_db_pool() 

    conn = None
    cursor = None
    try:
        conn = await db_pool.acquire()
        cursor = await conn.cursor()
        yield conn, cursor
    except Exception as e:
        if conn:
            await conn.rollback()
        raise e
    finally:
        if cursor:
            await cursor.close()
        if conn:
            await db_pool.release(conn)
