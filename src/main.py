from fastapi import FastAPI
from src.models import FeatureVectorChurn

app = FastAPI()


@app.get("/")
async def root(): return {"message": "ml churn service is running"}


@app.post("/predict")
async def predict(data: FeatureVectorChurn):
    return data