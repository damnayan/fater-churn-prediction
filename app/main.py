from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from app.schemas import (
    UserFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from app.model import get_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_model()
        print("[*] XGBoost Churn Model loaded successfully.")
    except Exception as e:
        print(f"[!] Warning during model load: {e}")
    yield


app = FastAPI(
    title="Fater Pampers Churn Prediction API",
    description="Production-ready inference service for Coccole Pampers churn risk scoring.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "service": "churn-inference-service"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
def predict_churn(user: UserFeatures):
    try:
        service = get_model()
        return service.predict_single(user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
)
def predict_churn_batch(payload: BatchPredictionRequest):
    try:
        service = get_model()
        results = [service.predict_single(u) for u in payload.users]
        return BatchPredictionResponse(predictions=results)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
