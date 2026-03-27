from fastapi import FastAPI
from sklearn.metrics import accuracy_score, f1_score

from src.dataset import load_dataset
from src.model import train_churn_model
from src.models import FeatureVectorChurn, DatasetRowChurn
from src.preprocessing import prepare_data

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


@app.get("/dataset/split-info")
async def split_info():
    df = load_dataset()
    x_train, x_test, y_train, y_test = prepare_data(df)
    return {
        "train_size": len(x_train),
        "test_size": len(y_test),
        "churn_train": y_train.value_counts().to_dict(),
        "churn_test": y_test.value_counts().to_dict()
    }


@app.post("/model/train")
async def train():
    df = load_dataset()
    X_train, X_test, y_train, y_test = prepare_data(df)
    trained_pipeline = train_churn_model(X_train, y_train)
    y_pred = trained_pipeline.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }