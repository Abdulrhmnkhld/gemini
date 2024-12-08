from fastapi import FastAPI
from app.api.v1.endpoints import router as api_router
from app.db.base import Base, engine

app = FastAPI()

Base.metadata.create_all(bind=engine)

app.include_router(api_router, prefix="/api/v1")
