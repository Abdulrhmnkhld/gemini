from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.models import User
from app.db.base import SessionLocal
from app.services.gemini_service import gemini_response
from app.utils.auth import create_access_token, verify_access_token
from pydantic import BaseModel

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# User schemas
class UserLogin(BaseModel):
    username: str
    password: str

class UserRegister(BaseModel):
    username: str
    password: str

class GeminiRequest(BaseModel):
    text: str

# Dependency to fetch the current user from a token
def get_current_user(token: str = Depends(verify_access_token), db: Session = Depends(get_db)):
    username = token
    db_user = db.query(User).filter(User.username == username).first()
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return db_user

# Registration endpoint
@router.post("/register/")
def register_user(user: UserRegister, db: Session = Depends(get_db)):
    # Check if the username already exists
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken.",
        )
    
    # Create and save the new user
    new_user = User(username=user.username)
    new_user.set_password(user.password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User registered successfully.", "username": new_user.username}

# Login endpoint
@router.post("/login/")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not db_user.verify_password(user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )
    # Generate a JWT token
    access_token = create_access_token(data={"sub": db_user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# Protected endpoint
@router.get("/protected/")
def protected_route(current_user: User = Depends(get_current_user)):
    return {"message": "You have access!", "username": current_user.username}

# Protected Gemini endpoint
@router.post("/gemini/")
def call_gemini(request: GeminiRequest, current_user: User = Depends(get_current_user)):
    try:
        response = gemini_response(request.text)
        return {"username": current_user.username, "input": request.text, "gemini_response": response}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to communicate with Gemini: {str(e)}",
        )
