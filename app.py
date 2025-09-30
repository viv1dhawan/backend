# app.py (Updated for pyodbc)
from fastapi import APIRouter, HTTPException, Depends, requests, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import List, Optional
import os
from together import Together
from fastapi.responses import RedirectResponse
import requests 
# NEW: Email sending imports
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Import database operations
import db_info.user_db as user_db
import db_info.application_db as application_db
import database

SENDER_EMAIL = os.getenv("EMAIL_SENDER", "dark.shadow.dhawan@gmail.com") # Replace with your sender email
SENDER_PASSWORD = os.getenv("EMAIL_PASSWORD", "sdoh iyfv djzb qemh") # Replace with your email password or app-specific password
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com") # Example for Gmail
SMTP_PORT = int(os.getenv("SMTP_PORT", 587)) # 587 for TLS, 465 for SSL

# Import Pydantic schemas
from Schema.user_schema import UserCreate, UserOut, PasswordReset, PasswordResetRequest, EmailVerification, UserUpdate, EmailVerificationRequest
# Import all new and existing application schemas
from Schema.application_schema import (
    HuggingFaceRequest,
    HuggingFaceResponse,
    QuestionInteractRequest,
    CommentAddRequest,
    QuestionCreate,
    QuestionInteractionResponse,
    CommentResponse,
    ResearcherResponse,
    QuestionResponse,
    QuestionsListResponse,
    ResearcherDeleteRequest,
    ResearcherUpdateRequest,
    ResearcherAddRequest,
    ResearcherInputRequest,
    QuestionInputRequest
)

# Token configuration
SECRET_KEY = "Iamvivekdhawan"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1 * 60

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="users/token")

# Routers
users_router = APIRouter()
app_router = APIRouter()
qna_router = APIRouter()
researcher_router = APIRouter()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email: str = payload.get("sub")
        if user_email is None:
            raise credentials_exception
        
        conn = database.get_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection error")
        
        user_in_db = user_db.get_user_by_email(conn, user_email)
        conn.close()
        
        if user_in_db is None:
            raise credentials_exception
            
        return user_in_db
    except JWTError:
        raise credentials_exception
    except Exception as e:
        print(f"Error getting current user: {e}")
        raise credentials_exception
        
def send_email(receiver_email: str, subject: str, body: str, html_body: str = None):
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = SENDER_EMAIL
    message["To"] = receiver_email

    # Turn these into plain/html MIMEText objects
    part1 = MIMEText(body, "plain")
    part2 = MIMEText(html_body, "html")

    # Add HTML part last so that email clients prefer the HTML version
    message.attach(part1)
    message.attach(part2)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls(context=context) # Secure the connection
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, receiver_email, message.as_string())
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False
    
def get_user_info(email: str):
    """
    Retrieve user information from the database by email.
    """
    conn = database.get_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection error")
    try:
        user = user_db.get_user_by_email(conn, email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    finally:
        conn.close()

@users_router.get("/me", response_model=UserOut)
def read_users_me(current_user: dict = Depends(get_current_user)):
    """
    Get the current authenticated user's information.

    This endpoint uses the 'get_current_user' dependency to
    validate the access token and retrieve the user's details from the database.
    """
    if current_user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    # Use get_user_info to fetch latest user info from DB
    user_info = get_user_info(current_user["email"])
    return user_info
    
@users_router.post("/register", response_model=UserOut)
def register_user(user: UserCreate):
    conn = database.get_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection error")
    
    try:
        existing_user = user_db.get_user_by_email(conn, user.email)
        if existing_user:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
        
        new_user = user_db.create_user(conn, user.dict())
        return new_user
    finally:
        conn.close()


@users_router.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = database.get_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection error")
        
    try:
        user = user_db.authenticate_user(conn, form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["email"]}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    finally:
        conn.close()

@users_router.post("/request-email-verification")
def request_email_verification(request_body: EmailVerificationRequest):
    conn = database.get_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection error")

    try:
        user = user_db.get_user_by_email(conn, request_body.email)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        if user["is_verified"]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already verified")

        token = user_db.create_email_verification_token(conn, request_body.email)
        
        verification_link = f"http://localhost:3000/verifyaccount?token={token}"
        
        subject = "Verify Your Email"
        body = f"""Hi {user["first_name"]},

Please click the link below to verify your email address:

{verification_link}

If you did not request this, please ignore this email.

Thanks,
The App Team
"""
        html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; background-color: #f4f4f7; margin: 0; padding: 0;">
                <table width="100%" style="max-width: 600px; margin: 20px auto; background: #ffffff; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);">
                <tr>
                    <td style="padding: 30px; text-align: center;">
                    <h2 style="color: #333;">Verify Your Email</h2>
                    <p style="color: #555;">Hi {user["first_name"]},</p>
                    <p style="color: #555; line-height: 1.5;">
                        Please confirm your email address by clicking the button below.
                    </p>
                    <a href="{verification_link}" 
                        style="display: inline-block; margin-top: 20px; padding: 12px 24px; 
                                background-color: #007BFF; color: #ffffff; text-decoration: none; 
                                border-radius: 6px; font-size: 16px; font-weight: bold;">
                        Verify Email
                    </a>
                    <p style="margin-top: 30px; font-size: 13px; color: #999;">
                        If you did not request this, you can safely ignore this email.
                    </p>
                    <p style="margin-top: 20px; color: #333; font-weight: bold;">The App Team</p>
                    </td>
                </tr>
                </table>
            </body>
            </html>
            """
        if send_email(request_body.email, subject, body, html_body):
            return {"message": "Verification email sent successfully."}
        else:
            raise HTTPException(status_code=500, detail="Failed to send verification email.")
    finally:
        conn.close()

@users_router.post("/verify-email")
def verify_email(input: EmailVerification):
    conn = database.get_connection()
    token = input.token
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection error")
    
    try:
        if user_db.verify_user_with_token(conn, token):
            # Redirect instead of returning plain string
            return {
                "message": "Email verified successfully. You can now log in.",
                "status": 200
            }
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired token.")
    finally:
        conn.close()
        
@users_router.post("/password-reset-request")
def password_reset_request(request: PasswordResetRequest):
    conn = database.get_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection error")
        
    try:
        user = user_db.get_user_by_email(conn, request.email)
        if not user:
            # Prevent user enumeration attacks by returning a generic success message
            return {"message": "If an account with that email exists, a password reset link has been sent."}

        token = user_db.create_password_reset_token(conn, request.email)
        
        reset_link = f"http://127.0.0.1:8000/users/reset-password?token={token}"
        
        subject = "Password Reset Request"
        body = f"""Hi {user["first_name"]},

You have requested a password reset. Please click the link below to reset your password:

{reset_link}

If you did not request this, please ignore this email.

Thanks,
The App Team
"""
        html_body = f"""
        <html>
            <body>
                <p>Hi {user["first_name"]},</p>
                <p>You have requested a password reset. Please click the link below to reset your password:</p>
                <a href="{reset_link}">Reset Password</a>
                <p>If you did not request this, please ignore this email.</p>
                <p>Thanks,<br>The App Team</p>
            </body>
        </html>
        """
        
        send_email(request.email, subject, body, html_body)
        return {"message": "If an account with that email exists, a password reset link has been sent."}
    finally:
        conn.close()

@users_router.post("/reset-password")
def reset_password(request: PasswordReset):
    conn = database.get_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection error")
    
    try:
        if user_db.reset_password_with_token(conn, request.token, request.new_password):
            return {"message": "Password has been reset successfully."}
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired token.")
    finally:
        conn.close()

@users_router.put("/update-user", response_model=UserOut)
def update_user(request: UserUpdate, current_user: dict = Depends(get_current_user)):
    conn = database.get_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection error")

    try:
        updated_user = user_db.update_user(conn, current_user["email"], request.dict(exclude_unset=True))
        if not updated_user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        return updated_user
    finally:
        conn.close()


# ============================================================
# Geophysical Data Router Endpoints
# ============================================================

HF_TOKEN = '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'
HF_API = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"

@app_router.post("/generate-ai-response", response_model=HuggingFaceResponse)
def get_hf_ai_response(request: HuggingFaceRequest, current_user: dict = Depends(get_current_user)):
    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {"inputs": request.prompt}

        response = requests.post(HF_API, headers=headers, json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        result = response.json()
        generated_text = result[0].get("generated_text", "")

        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error from Hugging Face API: {e}")


# ============================================================
# Q&A Router Endpoints
# ============================================================

@qna_router.post("/add_new_questions", response_model=QuestionResponse)
def create_question(question: QuestionCreate, current_user: dict = Depends(get_current_user)):
    conn = database.get_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection error")
    
    try:
        new_question = application_db.create_question_db(conn, current_user["id"], question.text)
        return new_question
    finally:
        conn.close()

@qna_router.get("/get_all_questions", response_model=QuestionsListResponse)
def get_questions_list():
    conn = database.get_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection error")
    
    try:
        questions = application_db.get_all_questions_with_details(conn)
        return {"questions": questions}
    finally:
        conn.close()


@qna_router.post("/questions/{question_id}", response_model=QuestionResponse)
def get_question(Input: QuestionInputRequest): 
    conn = database.get_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection error")
        
    try:
        question = application_db.get_question_by_id_db(conn, Input.question_id)
        if not question:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found")

        comments = application_db.get_comments_for_question_db(conn, Input.question_id)
        question["comments"] = comments
        
        return question
    finally:
        conn.close()

@qna_router.post("/questions/{question_id}/comments", response_model=CommentResponse)
def add_comment(comment: CommentAddRequest, current_user: dict = Depends(get_current_user)):
    conn = database.get_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection error")

    try:
        new_comment = application_db.add_comment_to_question_db(conn, comment.question_id, current_user["id"], comment.text)
        if not new_comment:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found")
        return new_comment
    finally:
        conn.close()

@qna_router.post("/questions/{question_id}/interact", response_model=QuestionInteractionResponse)
def interact_with_question(
    interaction: QuestionInteractRequest,
    current_user: dict = Depends(get_current_user),
):
    conn = database.get_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection error")

    try:
        # If enum, use its .value otherwise use the value directly
        interaction_type = getattr(interaction.type, "value", interaction.type)

        updated_interaction = application_db.add_or_update_question_interaction_db(
            conn, current_user["id"], interaction.question_id, interaction_type
        )

        # add_or_update returns None if the question doesn't exist
        if not updated_interaction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Question not found."
            )

        return updated_interaction
    finally:
        conn.close()

# ============================================================
# Researchers Router Endpoints
# ============================================================

@researcher_router.get("/", response_model=list[ResearcherResponse])
def get_researcher():
    conn = database.get_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection error")
        
    try:
        researchers = application_db.get_all_researchers_db(conn)
        if not researchers:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No researchers found")
        return researchers   # now FastAPI will validate as list[ResearcherResponse]
    finally:
        conn.close()

@researcher_router.post("/addresearcher", response_model=ResearcherResponse)
def add_researcher(
    request: ResearcherAddRequest,
    current_user: dict = Depends(get_current_user)
):
    conn = database.get_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection error")
    
    try:
        # Assuming user_id from the current_user is used, not the request body
        new_researcher = application_db.add_researcher_db(conn, request.dict(exclude={'user_id'}), current_user["id"])
        return new_researcher
    finally:
        conn.close()

@researcher_router.put("/updateresearcher", response_model=ResearcherResponse)
def update_researcher(
    request: ResearcherUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    conn = database.get_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection error")
        
    try:
        updated_researcher = application_db.update_researcher_db(conn, request.dict(), current_user["id"])
        if not updated_researcher:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to update this researcher entry or entry not found.")
        return updated_researcher
    finally:
        conn.close()

@researcher_router.delete("/deleteresearcher")
def delete_researcher(
    request: ResearcherDeleteRequest,
    current_user: dict = Depends(get_current_user)
):
    conn = database.get_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection error")
        
    try:
        deleted = application_db.delete_researcher_db(conn, request.researcher_id, current_user["id"])
        if not deleted:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to delete this researcher entry or entry not found.")
        return {"message": "Researcher entry deleted successfully."}
    finally:
        conn.close()
