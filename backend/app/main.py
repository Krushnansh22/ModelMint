"""ModelMint backend entrypoint."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.database.database import init_db
from app.api import models as models_api
from app.api import inference as inference_api
from app.api import monitoring as monitoring_api
from app.models.adapters.base import (
    AdapterDetectionError,
    AdapterLoadError,
    AdapterInferenceError,
)

app = FastAPI(title="ModelMint", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP: no auth/multi-tenancy yet, see build guide section 1
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.exception_handler(AdapterDetectionError)
def handle_detection_error(_request, exc: AdapterDetectionError):
    return JSONResponse(status_code=400, content={"detail": "Unsupported model format", "reason": str(exc)})


@app.exception_handler(AdapterLoadError)
def handle_load_error(_request, exc: AdapterLoadError):
    return JSONResponse(status_code=422, content={"detail": "Model failed to load", "reason": str(exc)})


@app.exception_handler(AdapterInferenceError)
def handle_inference_error(_request, exc: AdapterInferenceError):
    return JSONResponse(status_code=400, content={"detail": "Invalid input for this model", "reason": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(models_api.router)
app.include_router(inference_api.router)
app.include_router(monitoring_api.router)
