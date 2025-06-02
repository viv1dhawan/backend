# db_info/user_db.py
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import bcrypt
import secrets
import sqlalchemy

# Import database and models from the parent directory
from database import database
from models import users, password_reset_tokens, email_verification_tokens
# Import schema from the sibling Schema directory
from Schema.user_schema import UserCreate

# --- Password Hashing and Verification ---
def hash_password(password: str) -> str:
    """Hashes a plain-text password using bcrypt."""
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return hashed_password.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain-text password against a hashed password."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# --- User CRUD Operations ---
async def create_user(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a new user in the database.
    Args:
        user_data (dict): Dictionary containing user details (first_name, last_name, email, password).
    Returns:
        dict: The created user's data (excluding hashed password).
    """
    hashed_password = hash_password(user_data["password"])
    query = users.insert().values(
        first_name=user_data["first_name"],
        last_name=user_data["last_name"],
        email=user_data["email"],
        hashed_password=hashed_password,
        is_verified=False,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    user_id = await database.execute(query)
    # Fetch the newly created user to return all its data, including auto-generated fields
    return await get_user_by_id(user_id)

async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a user from the database by their email address.
    Args:
        email (str): The email of the user to retrieve.
    Returns:
        Optional[dict]: The user's data as a dictionary, or None if not found.
    """
    query = users.select().where(users.c.email == email)
    return await database.fetch_one(query)

async def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieves a user from the database by their ID.
    Args:
        user_id (int): The ID of the user to retrieve.
    Returns:
        Optional[dict]: The user's data as a dictionary, or None if not found.
    """
    query = users.select().where(users.c.id == user_id)
    return await database.fetch_one(query)

async def get_all_users() -> List[Dict[str, Any]]:
    """
    Retrieves all users from the database.
    Returns:
        List[dict]: A list of all user data.
    """
    query = users.select()
    return await database.fetch_all(query)

async def update_user_details(email: str, updated_fields: Dict[str, Any]) -> None:
    """
    Updates specific fields for a user.
    Args:
        email (str): The email of the user to update.
        updated_fields (dict): A dictionary of fields to update (e.g., {"first_name": "NewName"}).
    """
    query = users.update().where(users.c.email == email).values(**updated_fields, updated_at=datetime.utcnow())
    await database.execute(query)

# --- Password Reset Token Management ---
async def create_password_reset_token(email: str) -> str:
    """
    Generates and stores a password reset token for a given email.
    Args:
        email (str): The email for which to generate the token.
    Returns:
        str: The generated token.
    """
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=1)  # Token valid for 1 hour

    # Delete any existing tokens for this email to ensure only one active token
    delete_query = password_reset_tokens.delete().where(password_reset_tokens.c.email == email)
    await database.execute(delete_query)

    query = password_reset_tokens.insert().values(
        email=email,
        token=token,
        created_at=datetime.utcnow(),
        expires_at=expires_at
    )
    await database.execute(query)
    return token

async def verify_password_reset_token(token: str) -> Optional[str]:
    """
    Verifies a password reset token and returns the associated email if valid.
    Args:
        token (str): The password reset token to verify.
    Returns:
        Optional[str]: The email associated with the token, or None if invalid/expired.
    """
    query = password_reset_tokens.select().where(
        password_reset_tokens.c.token == token,
        password_reset_tokens.c.expires_at > datetime.utcnow()
    )
    token_record = await database.fetch_one(query)
    if token_record:
        # Invalidate the token after use
        delete_query = password_reset_tokens.delete().where(password_reset_tokens.c.token == token)
        await database.execute(delete_query)
        return token_record["email"]
    return None

async def update_password(email: str, new_password: str) -> None:
    """
    Updates a user's password after successful verification.
    Args:
        email (str): The email of the user whose password to update.
        new_password (str): The new plain-text password.
    """
    hashed_password = hash_password(new_password)
    query = users.update().where(users.c.email == email).values(hashed_password=hashed_password, updated_at=datetime.utcnow())
    await database.execute(query)

# --- Email Verification Token Management ---
async def generate_verification_token(email: str) -> str:
    """
    Generates and stores an email verification token for a given email.
    Args:
        email (str): The email for which to generate the token.
    Returns:
        str: The generated token.
    """
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=24)  # Token valid for 24 hours

    # Delete any existing unverified tokens for this email
    delete_query = email_verification_tokens.delete().where(email_verification_tokens.c.email == email)
    await database.execute(delete_query)

    query = email_verification_tokens.insert().values(
        email=email,
        token=token,
        created_at=datetime.utcnow(),
        expires_at=expires_at
    )
    await database.execute(query)
    return token

async def verify_user_with_token(token: str) -> bool:
    """
    Verifies a user's email using the provided token.
    Args:
        token (str): The email verification token.
    Returns:
        bool: True if verification is successful, False otherwise.
    """
    query = email_verification_tokens.select().where(
        email_verification_tokens.c.token == token,
        email_verification_tokens.c.expires_at > datetime.utcnow()
    )
    token_record = await database.fetch_one(query)

    if token_record:
        email = token_record["email"]
        # Update user's is_verified status
        update_query = users.update().where(users.c.email == email).values(is_verified=True, updated_at=datetime.utcnow())
        await database.execute(update_query)

        # Invalidate the token after use
        delete_query = email_verification_tokens.delete().where(email_verification_tokens.c.token == token)
        await database.execute(delete_query)
        return True
    return False
