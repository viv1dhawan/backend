from pydantic import BaseModel, EmailStr
from typing import Optional

# Pydantic model for creating a new user
class UserCreate(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str

class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    new_password: Optional[str] = None

# Pydantic model for user output (excluding sensitive info like hashed password)
class UserOut(BaseModel):
    id: int
    first_name: str
    last_name: str
    email: EmailStr
    is_verified: bool

# Pydantic model for password reset request (email)
class PasswordResetRequest(BaseModel):
    email: EmailStr

# Pydantic model for actual password reset (token and new password)
class PasswordReset(BaseModel):
    token: str
    new_password: str

# Pydantic model for requesting an email verification token
class EmailVerificationRequest(BaseModel):
    email: EmailStr

# Pydantic model for verifying email with a token
class EmailVerification(BaseModel):
    token: str
