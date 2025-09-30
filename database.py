# database.py - MySQL connection and table creation with retry
import time
import mysql.connector
from mysql.connector import Error
from models import TABLES

# MySQL connection parameters (from Railway)
BASE_DB_CONFIG = {
    "host": "centerbeam.proxy.rlwy.net",
    "port": 36360,
    "user": "root",
    "password": "FuWjJBjPXkEWuNviJTZHLkOmWsgAhvdR",
    "database": "railway"
}

def get_connection(max_retries=3, delay=2):
    """
    Open and return a new connection to MySQL.
    Tries normal connection first, falls back to ssl_disabled=True if handshake fails.
    Retries up to `max_retries` times with `delay` seconds between attempts.
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt}: connecting to MySQL (normal)...")
            conn = mysql.connector.connect(**BASE_DB_CONFIG)
            if conn.is_connected():
                print("Connected to MySQL successfully (normal mode).")
                return conn
        except Error as ex:
            print(f"Normal connection failed: {ex}")
            if attempt < max_retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)

    # If normal connection fails, try SSL disabled
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt}: connecting to MySQL with ssl_disabled=True...")
            config = BASE_DB_CONFIG.copy()
            config["ssl_disabled"] = True
            conn = mysql.connector.connect(**config)
            if conn.is_connected():
                print("Connected to MySQL successfully (ssl_disabled mode).")
                return conn
        except Error as ex2:
            print(f"SSL-disabled connection failed: {ex2}")
            if attempt < max_retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)

    print("All connection attempts failed.")
    return None


def create_tables():
    """
    Creates tables if they do not exist.
    """
    conn = get_connection()
    if not conn:
        print("Failed to get database connection to create tables.")
        return

    try:
        cursor = conn.cursor()
        for table_sql in TABLES:
            cursor.execute(table_sql)
            print("Table checked/created.")
        conn.commit()
    except Error as ex:
        print(f"Error creating tables: {ex}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn and conn.is_connected():
            conn.close()


if __name__ == "__main__":
    create_tables()
