# database.py - MySQL connection and table creation with retry
import mysql.connector
from mysql.connector import Error
from models import TABLES

# MySQL connection parameters (from Railway)
BASE_DB_CONFIG = {
    "host": "shortline.proxy.rlwy.net",
    "port": 52270,
    "user": "root",
    "password": "EYDIQLYoErENIIPBMucXrLEHRJmtuEVO",
    "database": "railway"
}


def get_connection():
    """
    Open and return a new connection to MySQL.
    Tries normal connection first, falls back to ssl_disabled=True if handshake fails.
    """
    try:
        print("Trying normal MySQL connection...")
        conn = mysql.connector.connect(**BASE_DB_CONFIG)
        if conn.is_connected():
            print("✅ Connected to MySQL successfully (normal mode).")
            return conn
    except Error as ex:
        print(f"Normal connection failed: {ex}")

        # Retry with SSL disabled
        try:
            print("Retrying MySQL connection with ssl_disabled=True...")
            config = BASE_DB_CONFIG.copy()
            config["ssl_disabled"] = True
            conn = mysql.connector.connect(**config)
            if conn.is_connected():
                print("✅ Connected to MySQL successfully (ssl_disabled mode).")
                return conn
        except Error as ex2:
            print(f"SSL-disabled connection failed: {ex2}")
            return None


def create_tables():
    """
    Creates tables if they do not exist.
    """
    try:
        conn = get_connection()
        if not conn:
            print("Failed to get database connection to create tables.")
            return

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
