# database.py - Updated for MySQL
import mysql.connector
from mysql.connector import Error
from datetime import datetime

# MySQL connection parameters (parsed from your connection string)
DB_CONFIG = {
    "host": "shortline.proxy.rlwy.net",
    "port": 52270,
    "user": "root",
    "password": "EYDIQLYoErENIIPBMucXrLEHRJmtuEVO",
    "database": "railway"
}

def get_connection():
    """
    Open and return a new connection to MySQL.
    Caller is responsible for closing the connection.
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            return conn
    except Error as ex:
        print(f"Database connection error: {ex}")
        return None

def create_tables():
    """
    Creates tables if they do not exist. 
    This is a synchronous operation.
    """
    try:
        conn = get_connection()
        if not conn:
            print("Failed to get database connection to create tables.")
            return

        cursor = conn.cursor()

        # SQL queries for table creation (MySQL syntax)
        create_users_table = """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            email VARCHAR(255) NOT NULL UNIQUE,
            hashed_password VARCHAR(255) NOT NULL,
            first_name VARCHAR(255),
            last_name VARCHAR(255),
            is_verified BOOLEAN DEFAULT 0 NOT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        );
        """

        create_password_reset_tokens_table = """
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            id INT AUTO_INCREMENT PRIMARY KEY,
            email VARCHAR(255) NOT NULL,
            token VARCHAR(255) NOT NULL UNIQUE,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME NOT NULL
        );
        """

        create_email_verification_tokens_table = """
        CREATE TABLE IF NOT EXISTS email_verification_tokens (
            id INT AUTO_INCREMENT PRIMARY KEY,
            email VARCHAR(255) NOT NULL,
            token VARCHAR(255) NOT NULL UNIQUE,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME NOT NULL
        );
        """

        create_questions_table = """
        CREATE TABLE IF NOT EXISTS questions (
            id CHAR(36) PRIMARY KEY,
            user_id INT,
            text TEXT NOT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """

        create_comments_table = """
        CREATE TABLE IF NOT EXISTS comments (
            id CHAR(36) PRIMARY KEY,
            user_id INT,
            question_id CHAR(36),
            text TEXT NOT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE
        );
        """

        create_question_interactions_table = """
        CREATE TABLE IF NOT EXISTS question_interactions (
            id CHAR(36) PRIMARY KEY,
            user_id INT,
            question_id CHAR(36),
            type VARCHAR(10) NOT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (user_id, question_id),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE
        );
        """

        create_researchers_table = """
        CREATE TABLE IF NOT EXISTS researchers (
            id INT AUTO_INCREMENT PRIMARY KEY,
            title VARCHAR(255) NOT NULL,
            authors VARCHAR(500) NOT NULL,
            user_id INT,
            profile VARCHAR(500) NOT NULL,
            publication_date VARCHAR(50) NOT NULL,
            url VARCHAR(1000) NOT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """

        tables = [
            create_users_table,
            create_password_reset_tokens_table,
            create_email_verification_tokens_table,
            create_questions_table,
            create_comments_table,
            create_question_interactions_table,
            create_researchers_table,
        ]

        for table_sql in tables:
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
