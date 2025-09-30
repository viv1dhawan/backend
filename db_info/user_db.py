# db_info/user_db.py - Updated for pyodbc
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import bcrypt
import secrets
import pyodbc

# Helper function to convert pyodbc.Row to a dictionary
def row_to_dict(row):
    if not row:
        return None
    return {column[0]: row[i] for i, column in enumerate(row.cursor_description)}

# --- Password Hashing and Verification ---
def hash_password(password: str) -> str:
    """Hashes a plain-text password using bcrypt."""
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return hashed_password.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain-text password against a hashed password."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# --- User CRUD Operations ---
def create_user(conn, user_data: dict):
    cursor = conn.cursor()
    try:
        hashed_password = hash_password(user_data["password"])
        query = """
        INSERT INTO users (first_name, last_name, email, hashed_password)
        OUTPUT INSERTED.id, INSERTED.first_name, INSERTED.last_name, INSERTED.email, INSERTED.is_verified, INSERTED.created_at, INSERTED.updated_at
        VALUES (?, ?, ?, ?);
        """
        cursor.execute(query, (user_data["first_name"], user_data["last_name"], user_data["email"], hashed_password))
        
        # Fetch the newly created user record
        new_user_row = cursor.fetchone()
        conn.commit()
        return row_to_dict(new_user_row)
    finally:
        cursor.close()

def get_user_by_email(conn, email: str):
    """Retrieves a single user by their email address."""
    cursor = conn.cursor()
    try:
        query = "SELECT id, first_name, last_name, email, hashed_password, is_verified FROM users WHERE email = ?;"
        cursor.execute(query, (email,))
        user_row = cursor.fetchone()
        return row_to_dict(user_row)
    finally:
        cursor.close()

def get_user_by_id(conn, user_id: int):
    """Retrieves a single user by their ID."""
    cursor = conn.cursor()
    try:
        query = "SELECT id, first_name, last_name, email, is_verified FROM users WHERE id = ?;"
        cursor.execute(query, (user_id,))
        user_row = cursor.fetchone()
        return row_to_dict(user_row)
    finally:
        cursor.close()

def authenticate_user(conn, email: str, password: str):
    """Authenticates a user by email and password."""
    user = get_user_by_email(conn, email)
    if user and verify_password(password, user["hashed_password"]):
        return user
    return None

def update_user(conn, email: str, user_data: dict):
    """Updates a user's details."""
    cursor = conn.cursor()
    try:
        set_clauses = []
        params = []
        
        if "first_name" in user_data:
            set_clauses.append("first_name = ?")
            params.append(user_data["first_name"])
        if "last_name" in user_data:
            set_clauses.append("last_name = ?")
            params.append(user_data["last_name"])
        if "new_password" in user_data:
            hashed_password = hash_password(user_data["new_password"])
            set_clauses.append("hashed_password = ?")
            params.append(hashed_password)
            
        if not set_clauses:
            return get_user_by_email(conn, email)
            
        set_clauses.append("updated_at = GETDATE()")
        
        query = f"UPDATE users SET {', '.join(set_clauses)} WHERE email = ?;"
        params.append(email)
        
        cursor.execute(query, params)
        conn.commit()
        return get_user_by_email(conn, email)
    finally:
        cursor.close()

# --- Password Reset Token Management ---
def create_password_reset_token(conn, email: str):
    """Creates and stores a password reset token for the given email."""
    cursor = conn.cursor()
    try:
        token = secrets.token_urlsafe(32)
        # Token expires in 1 hour
        expires_at = datetime.utcnow() + timedelta(hours=1)
        
        # Delete any existing tokens for this email
        delete_query = "DELETE FROM password_reset_tokens WHERE email = ?;"
        cursor.execute(delete_query, (email,))
        
        insert_query = "INSERT INTO password_reset_tokens (email, token, created_at, expires_at) VALUES (?, ?, GETDATE(), ?);"
        cursor.execute(insert_query, (email, token, expires_at))
        conn.commit()
        return token
    finally:
        cursor.close()

def reset_password_with_token(conn, token: str, new_password: str):
    """Resets a user's password using a valid token."""
    cursor = conn.cursor()
    try:
        # Check if the token is valid and not expired
        query = "SELECT email FROM password_reset_tokens WHERE token = ? AND expires_at > GETDATE();"
        cursor.execute(query, (token,))
        token_record = cursor.fetchone()
        
        if token_record:
            email = token_record[0]
            hashed_password = hash_password(new_password)
            
            # Update user's password
            update_query = "UPDATE users SET hashed_password = ?, updated_at = GETDATE() WHERE email = ?;"
            cursor.execute(update_query, (hashed_password, email))
            
            # Delete the used token
            delete_query = "DELETE FROM password_reset_tokens WHERE email = ?;"
            cursor.execute(delete_query, (email,))
            
            conn.commit()
            return True
    finally:
        cursor.close()
    return False

# --- Email Verification Token Management ---
def create_email_verification_token(conn, email: str):
    """Creates and stores an email verification token for a user."""
    cursor = conn.cursor()
    try:
        # Generate a secure, unique token
        token = secrets.token_urlsafe(32)
        # Token expires in 24 hours
        expires_at = datetime.utcnow() + timedelta(hours=24)
        
        # Delete any existing tokens for this email to prevent multiple valid tokens
        delete_query = "DELETE FROM email_verification_tokens WHERE email = ?;"
        cursor.execute(delete_query, (email,))
        conn.commit()
        
        # Insert the new token
        insert_query = "INSERT INTO email_verification_tokens (email, token, created_at, expires_at) VALUES (?, ?, ?, ?);"
        cursor.execute(insert_query, (email, token, datetime.utcnow(), expires_at))
        conn.commit()
        return token
    finally:
        cursor.close()

def verify_user_with_token(conn, token: str):
    """
    Verifies a user's email using the provided token.
    """
    cursor = conn.cursor()
    try:
        query = "SELECT email FROM email_verification_tokens WHERE token = ? AND expires_at > GETDATE()"
        cursor.execute(query, (token,))
        token_record = cursor.fetchone()

        if token_record:
            email = token_record[0]
            # Update user's is_verified status
            update_query = "UPDATE users SET is_verified = 1, updated_at = GETDATE() WHERE email = ?;"
            cursor.execute(update_query, (email,))
            
            # Delete the used token
            delete_query = "DELETE FROM email_verification_tokens WHERE email = ?;"
            cursor.execute(delete_query, (email,))
            
            conn.commit()
            return True
    except pyodbc.Error as ex:
        print(f"Database error during email verification: {ex}")
        conn.rollback() # Rollback changes if an error occurred
    finally:
        cursor.close()
    return False

def is_email_verified(conn, email: str):
    """
    Checks if the user's email is verified.
    """
    cursor = conn.cursor()
    try:
        query = "SELECT is_verified FROM users WHERE email = ?;"
        cursor.execute(query, (email,))
        is_verified_row = cursor.fetchone()
        if is_verified_row:
            return bool(is_verified_row[0]) # Convert from SQL boolean (0/1) to Python bool
    finally:
        cursor.close()
    return False
