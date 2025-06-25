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
        is_verified=False, # New users are unverified by default
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    last_record_id = await database.execute(query)
    # Fetch the newly created user to return it
    return await get_user_by_id(last_record_id)


async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a user by their email address.
    """
    query = users.select().where(users.c.email == email)
    user_record = await database.fetch_one(query)
    return dict(user_record) if user_record else None

async def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieves a user by their ID.
    """
    query = users.select().where(users.c.id == user_id)
    user_record = await database.fetch_one(query)
    return dict(user_record) if user_record else None

async def update_user_details(email: str, updated_fields: Dict[str, Any]) -> None:
    """
    Updates specified fields for a user based on their email.
    Does not update password directly (use update_password for that).
    """
    # Ensure 'new_password' is not in updated_fields if it was passed
    updated_fields.pop("new_password", None)

    if not updated_fields: # If no fields to update after popping password
        return

    updated_fields["updated_at"] = datetime.utcnow() # Update timestamp

    query = users.update().where(users.c.email == email).values(**updated_fields)
    await database.execute(query)

async def update_password(email: str, new_password: str) -> None:
    """
    Updates a user's password.
    """
    hashed_password = hash_password(new_password)
    query = users.update().where(users.c.email == email).values(
        hashed_password=hashed_password,
        updated_at=datetime.utcnow()
    )
    await database.execute(query)

async def get_all_users() -> List[Dict[str, Any]]:
    """
    Retrieves all users from the database.
    """
    query = users.select()
    user_records = await database.fetch_all(query)
    return [dict(record) for record in user_records]

# --- Password Reset Token Operations ---
async def create_password_reset_token(email: str) -> str:
    """
    Generates and stores a password reset token for the given email.
    """
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=1)  # Token valid for 1 hour

    # Invalidate any existing tokens for this user
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
    """
    query = password_reset_tokens.select().where(
        password_reset_tokens.c.token == token,
        password_reset_tokens.c.expires_at > datetime.utcnow()
    )
    token_record = await database.fetch_one(query)

    if token_record:
        email = token_record["email"]
        # Invalidate the token after successful verification/use
        delete_query = password_reset_tokens.delete().where(password_reset_tokens.c.id == token_record["id"])
        await database.execute(delete_query)
        return email
    return None

# --- Email Verification Operations ---
async def generate_verification_token(email: str) -> str:
    """
    Generates and stores an email verification token.
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

        # Invalidate the token
        delete_query = email_verification_tokens.delete().where(email_verification_tokens.c.id == token_record["id"])
        await database.execute(delete_query)
        return True
    return False
