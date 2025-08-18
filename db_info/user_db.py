# db_info/user_db.py
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import bcrypt
import secrets
import aiomysql # Import aiomysql

# Note: models.py is no longer used for table definitions with raw SQL
# from models import users, password_reset_tokens, email_verification_tokens
# No longer import schema from Schema/user_schema as we are directly returning dicts
# from Schema.user_schema import UserCreate

# --- Password Hashing and Verification ---
def hash_password(password: str) -> str:
    """Hashes a plain-text password using bcrypt."""
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return hashed_password.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain-text password against a hashed password."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# --- User CRUD Operations ---
async def create_user(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a new user in the database.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        user_data (dict): Dictionary containing user details (first_name, last_name, email, password).
    Returns:
        dict: The created user's data (excluding hashed password).
    """
    hashed_password = hash_password(user_data["password"])
    query = """
    INSERT INTO users (first_name, last_name, email, hashed_password, is_verified, created_at, updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    params = (
        user_data["first_name"],
        user_data["last_name"],
        user_data["email"],
        hashed_password,
        False,
        datetime.utcnow(),
        datetime.utcnow()
    )
    await cursor.execute(query, params)
    await conn.commit()

    # Get the last inserted ID
    await cursor.execute("SELECT LAST_INSERT_ID() as id")
    result = await cursor.fetchone()
    last_record_id = result['id'] if result else None

    if not last_record_id:
        raise Exception("Failed to retrieve new user ID after insertion.")

    # Fetch the newly created user to return it
    return await get_user_by_id(conn, cursor, last_record_id)


async def get_user_by_email(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, email: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a user by their email address.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        email (str): The email address of the user.
    Returns:
        Optional[Dict[str, Any]]: The user's data or None if not found.
    """
    query = "SELECT id, email, hashed_password, first_name, last_name, is_verified, created_at, updated_at FROM users WHERE email = %s"
    await cursor.execute(query, (email,))
    user_record = await cursor.fetchone()
    return user_record

async def get_user_by_id(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, user_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieves a user by their ID.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        user_id (int): The ID of the user.
    Returns:
        Optional[Dict[str, Any]]: The user's data or None if not found.
    """
    query = "SELECT id, email, hashed_password, first_name, last_name, is_verified, created_at, updated_at FROM users WHERE id = %s"
    await cursor.execute(query, (user_id,))
    user_record = await cursor.fetchone()
    return user_record

async def update_user_details(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, email: str, updated_fields: Dict[str, Any]) -> None:
    """
    Updates specified fields for a user based on their email.
    Does not update password directly (use update_password for that).
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        email (str): The email of the user to update.
        updated_fields (Dict[str, Any]): Dictionary of fields to update.
    """
    updated_fields.pop("new_password", None) # Ensure 'new_password' is not in updated_fields

    if not updated_fields:
        return

    updated_fields["updated_at"] = datetime.utcnow() # Update timestamp

    # Construct the SET part of the SQL query dynamically
    set_clauses = []
    params = []
    for field, value in updated_fields.items():
        set_clauses.append(f"{field} = %s")
        params.append(value)

    if not set_clauses:
        return # No fields to update

    params.append(email) # Add email for the WHERE clause

    query = f"UPDATE users SET {', '.join(set_clauses)} WHERE email = %s"
    await cursor.execute(query, params)
    await conn.commit()

async def update_password(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, email: str, new_password: str) -> None:
    """
    Updates a user's password.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        email (str): The email of the user.
        new_password (str): The new plain-text password.
    """
    hashed_password = hash_password(new_password)
    query = "UPDATE users SET hashed_password = %s, updated_at = %s WHERE email = %s"
    params = (hashed_password, datetime.utcnow(), email)
    await cursor.execute(query, params)
    await conn.commit()

async def get_all_users(conn: aiomysql.Connection, cursor: aiomysql.DictCursor) -> List[Dict[str, Any]]:
    """
    Retrieves all users from the database.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
    Returns:
        List[Dict[str, Any]]: A list of all user data.
    """
    query = "SELECT id, email, hashed_password, first_name, last_name, is_verified, created_at, updated_at FROM users"
    await cursor.execute(query)
    user_records = await cursor.fetchall()
    return user_records

# --- Password Reset Token Operations ---
async def create_password_reset_token(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, email: str) -> str:
    """
    Generates and stores a password reset token for the given email.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        email (str): The email address for which to create the token.
    Returns:
        str: The generated token.
    """
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=1)  # Token valid for 1 hour

    # Invalidate any existing tokens for this user
    delete_query = "DELETE FROM password_reset_tokens WHERE email = %s"
    await cursor.execute(delete_query, (email,))

    insert_query = """
    INSERT INTO password_reset_tokens (email, token, created_at, expires_at)
    VALUES (%s, %s, %s, %s)
    """
    params = (email, token, datetime.utcnow(), expires_at)
    await cursor.execute(insert_query, params)
    await conn.commit()
    return token

async def verify_password_reset_token(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, token: str) -> Optional[str]:
    """
    Verifies a password reset token and returns the associated email if valid.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        token (str): The token to verify.
    Returns:
        Optional[str]: The email associated with the token, or None if invalid/expired.
    """
    query = "SELECT id, email FROM password_reset_tokens WHERE token = %s AND expires_at > %s"
    await cursor.execute(query, (token, datetime.utcnow()))
    token_record = await cursor.fetchone()

    if token_record:
        email = token_record["email"]
        # Invalidate the token after successful verification/use
        delete_query = "DELETE FROM password_reset_tokens WHERE id = %s"
        await cursor.execute(delete_query, (token_record["id"],))
        await conn.commit()
        return email
    return None

# --- Email Verification Operations ---
async def generate_verification_token(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, email: str) -> str:
    """
    Generates and stores an email verification token.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        email (str): The email address for which to generate the token.
    Returns:
        str: The generated token.
    """
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=24)  # Token valid for 24 hours

    # Delete any existing unverified tokens for this email
    delete_query = "DELETE FROM email_verification_tokens WHERE email = %s"
    await cursor.execute(delete_query, (email,))

    insert_query = """
    INSERT INTO email_verification_tokens (email, token, created_at, expires_at)
    VALUES (%s, %s, %s, %s)
    """
    params = (email, token, datetime.utcnow(), expires_at)
    await cursor.execute(insert_query, params)
    await conn.commit()
    return token

async def verify_user_with_token(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, token: str) -> bool:
    """
    Verifies a user's email using the provided token.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        token (str): The email verification token.
    Returns:
        bool: True if verification is successful, False otherwise.
    """
    query = "SELECT id, email FROM email_verification_tokens WHERE token = %s AND expires_at > %s"
    await cursor.execute(query, (token, datetime.utcnow()))
    token_record = await cursor.fetchone()

    if token_record:
        email = token_record["email"]
        # Update user's is_verified status
        update_query = "UPDATE users SET is_verified = TRUE, updated_at = %s WHERE email = %s"
        await cursor.execute(update_query, (datetime.utcnow(), email))

        # Invalidate the token
        delete_query = "DELETE FROM email_verification_tokens WHERE id = %s"
        await cursor.execute(delete_query, (token_record["id"],))
        await conn.commit()
        return True
    return False
