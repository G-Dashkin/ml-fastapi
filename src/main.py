from fastapi import FastAPI

from src.dataset import load_dataset
from src.models import FeatureVectorChurn, DatasetRowChurn

app = FastAPI()


@app.get("/")
async def root(): return {"message": "ml churn service is running"}


@app.post("/predict")
async def predict(data: FeatureVectorChurn):
    return data


@app.get("/dataset/preview")
async def preview(n: int = 5):
    df = load_dataset()
    rows = df.head(n).to_dict(orient="records")
    return [DatasetRowChurn(**row) for row in rows]


@app.get("/dataset/info")
async def info():
    df = load_dataset()
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "churn_distribution": df["churn"].value_counts().to_dict()
    }