import os
import subprocess

def run_migrations():
    print("Applying database migrations...")
    # result = subprocess.run(["alembic", "upgrade", "head"], check=True)
    # print(result.stdout)

def start_app():
    print("Starting the FastAPI app...")
    os.execvp("uvicorn", ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"])

if __name__ == "__main__":
    run_migrations()
    start_app()
