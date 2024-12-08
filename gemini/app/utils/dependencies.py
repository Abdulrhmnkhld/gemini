from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.models import User
from app.db.base import SessionLocal
from app.utils.auth import verify_access_token

def get_current_user(
    username: str = Depends(verify_access_token), db: Session = Depends(SessionLocal)
):
    db_user = db.query(User).filter(User.username == username).first()
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return db_user
