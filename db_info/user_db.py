# db_info/user_db.py
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import bcrypt
import secrets
import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, insert, delete

# Import models from the parent directory
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
async def create_user(session: AsyncSession, user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a new user in the database.
    Args:
        session (AsyncSession): The database session.
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
    result = await session.execute(query)
    await session.commit()
    
    # For MySQL, inserted_primary_key typically works if the ID is auto-incrementing
    last_record_id = result.inserted_primary_key[0] if result.inserted_primary_key else None
    if not last_record_id:
        raise Exception("Failed to retrieve new user ID after insertion.")
    
    # Fetch the newly created user to return it
    return await get_user_by_id(session, last_record_id)


async def get_user_by_email(session: AsyncSession, email: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a user by their email address.
    Args:
        session (AsyncSession): The database session.
        email (str): The email address of the user.
    Returns:
        Optional[Dict[str, Any]]: The user's data or None if not found.
    """
    query = users.select().where(users.c.email == email)
    result = await session.execute(query)
    user_record = result.first()
    return dict(user_record) if user_record else None

async def get_user_by_id(session: AsyncSession, user_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieves a user by their ID.
    Args:
        session (AsyncSession): The database session.
        user_id (int): The ID of the user.
    Returns:
        Optional[Dict[str, Any]]: The user's data or None if not found.
    """
    query = users.select().where(users.c.id == user_id)
    result = await session.execute(query)
    user_record = result.first()
    return dict(user_record) if user_record else None

async def update_user_details(session: AsyncSession, email: str, updated_fields: Dict[str, Any]) -> None:
    """
    Updates specified fields for a user based on their email.
    Does not update password directly (use update_password for that).
    Args:
        session (AsyncSession): The database session.
        email (str): The email of the user to update.
        updated_fields (Dict[str, Any]): Dictionary of fields to update.
    """
    # Ensure 'new_password' is not in updated_fields if it was passed
    updated_fields.pop("new_password", None)

    if not updated_fields: # If no fields to update after popping password
        return

    updated_fields["updated_at"] = datetime.utcnow() # Update timestamp

    query = users.update().where(users.c.email == email).values(**updated_fields)
    await session.execute(query)
    await session.commit()

async def update_password(session: AsyncSession, email: str, new_password: str) -> None:
    """
    Updates a user's password.
    Args:
        session (AsyncSession): The database session.
        email (str): The email of the user.
        new_password (str): The new plain-text password.
    """
    hashed_password = hash_password(new_password)
    query = users.update().where(users.c.email == email).values(
        hashed_password=hashed_password,
        updated_at=datetime.utcnow()
    )
    await session.execute(query)
    await session.commit()

async def get_all_users(session: AsyncSession) -> List[Dict[str, Any]]:
    """
    Retrieves all users from the database.
    Args:
        session (AsyncSession): The database session.
    Returns:
        List[Dict[str, Any]]: A list of all user data.
    """
    query = users.select()
    result = await session.execute(query)
    user_records = result.fetchall()
    return [dict(record) for record in user_records]

# --- Password Reset Token Operations ---
async def create_password_reset_token(session: AsyncSession, email: str) -> str:
    """
    Generates and stores a password reset token for the given email.
    Args:
        session (AsyncSession): The database session.
        email (str): The email address for which to create the token.
    Returns:
        str: The generated token.
    """
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=1)  # Token valid for 1 hour

    # Invalidate any existing tokens for this user
    delete_query = password_reset_tokens.delete().where(password_reset_tokens.c.email == email)
    await session.execute(delete_query)

    query = password_reset_tokens.insert().values(
        email=email,
        token=token,
        created_at=datetime.utcnow(),
        expires_at=expires_at
    )
    await session.execute(query)
    await session.commit()
    return token

async def verify_password_reset_token(session: AsyncSession, token: str) -> Optional[str]:
    """
    Verifies a password reset token and returns the associated email if valid.
    Args:
        session (AsyncSession): The database session.
        token (str): The token to verify.
    Returns:
        Optional[str]: The email associated with the token, or None if invalid/expired.
    """
    query = password_reset_tokens.select().where(
        password_reset_tokens.c.token == token,
        password_reset_tokens.c.expires_at > datetime.utcnow()
    )
    result = await session.execute(query)
    token_record = result.first()

    if token_record:
        email = token_record["email"]
        # Invalidate the token after successful verification/use
        delete_query = password_reset_tokens.delete().where(password_reset_tokens.c.id == token_record["id"])
        await session.execute(delete_query)
        await session.commit()
        return email
    return None

# --- Email Verification Operations ---
async def generate_verification_token(session: AsyncSession, email: str) -> str:
    """
    Generates and stores an email verification token.
    Args:
        session (AsyncSession): The database session.
        email (str): The email address for which to generate the token.
    Returns:
        str: The generated token.
    """
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=24)  # Token valid for 24 hours

    # Delete any existing unverified tokens for this email
    delete_query = email_verification_tokens.delete().where(email_verification_tokens.c.email == email)
    await session.execute(delete_query)

    query = email_verification_tokens.insert().values(
        email=email,
        token=token,
        created_at=datetime.utcnow(),
        expires_at=expires_at
    )
    await session.execute(query)
    await session.commit()
    return token

async def verify_user_with_token(session: AsyncSession, token: str) -> bool:
    """
    Verifies a user's email using the provided token.
    Args:
        session (AsyncSession): The database session.
        token (str): The email verification token.
    Returns:
        bool: True if verification is successful, False otherwise.
    """
    query = email_verification_tokens.select().where(
        email_verification_tokens.c.token == token,
        email_verification_tokens.c.expires_at > datetime.utcnow()
    )
    result = await session.execute(query)
    token_record = result.first()

    if token_record:
        email = token_record["email"]
        # Update user's is_verified status
        update_query = users.update().where(users.c.email == email).values(is_verified=True, updated_at=datetime.utcnow())
        await session.execute(update_query)

        # Invalidate the token
        delete_query = email_verification_tokens.delete().where(email_verification_tokens.c.id == token_record["id"])
        await session.execute(delete_query)
        await session.commit()
        return True
    return False
