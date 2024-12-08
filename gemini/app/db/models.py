# app/db/models.py
from sqlalchemy import Column, Integer, String
from app.db.base import Base
from passlib.context import CryptContext

# Password hashing context
pwd_context = CryptContext(schemes=["argon2","bcrypt"], deprecated="auto")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    api_key = Column(String, unique=True, index=True)

    # Hash password
    def set_password(self, password: str):
        self.hashed_password = pwd_context.hash(password)

    # Verify password
    def verify_password(self, password: str) -> bool:
        return pwd_context.verify(password, self.hashed_password)
