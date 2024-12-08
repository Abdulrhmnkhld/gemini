import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY")
    GEMINI_MODEL =  'gemini-1.5-pro'
settings = Settings()
