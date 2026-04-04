from app.routers import analysis
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from app.services.legal_engine import load_reasoning_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_on_startup = os.getenv("PRELOAD_REASONING_MODEL", "false").lower() == "true"
    print("System Startup")
    if preload_on_startup:
        print("Pre-loading reasoning model (PRELOAD_REASONING_MODEL=true)...")
        try:
            load_reasoning_model()
            print("Reasoning model preloaded successfully")
        except Exception as e:
            # Keep API available even if preload fails; model will lazy-load on demand.
            print(f"Reasoning model preload failed: {e}")
            print("Continuing startup with lazy model loading")
    else:
        print("Skipping model preload; reasoning model will load on first analysis request")
    yield
    print("System Shutdown")

app = FastAPI(title="Virtual Senior Prosecutor API", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Server is running. Go to /docs to test the API."}