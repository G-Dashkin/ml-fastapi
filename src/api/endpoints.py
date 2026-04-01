from datetime import datetime
from fastapi import APIRouter, HTTPException
from sklearn.metrics import accuracy_score, f1_score
from src.ml.dataset import load_dataset
from src.ml.model import train_churn_model, save_churn_model, load_churn_model, save_history, load_history
from src.ml.preprocessing import prepare_data, NUMERIC_COLS, CATEGORICAL_COLS
from src.schemas.churn_models import FeatureVectorChurn, DatasetRowChurn, PredictionResponseChurn, TrainingConfigChurn
from typing import Union, List
import pandas as pd
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

loaded_model = load_churn_model()


@router.get("/")
async def root(): return {"message": "ml churn service is running"}


@router.post("/predict")
async def predict(data: Union[FeatureVectorChurn, List[FeatureVectorChurn]]):
    if loaded_model is None: raise HTTPException(status_code=400, detail="Модель не обучена. Сначала переходим на /model/train")
    items = data if isinstance(data, list) else [data]
    df = pd.DataFrame([item.model_dump() for item in items], columns=NUMERIC_COLS + CATEGORICAL_COLS)
    predictions = loaded_model["pipeline"].predict(df)
    probabilities = loaded_model["pipeline"].predict_proba(df)[:, 1]
    results = [
        PredictionResponseChurn(
            churn=int(p.item()),
            probability=round(float(prob), 4)
        )
        for p, prob in zip(predictions, probabilities)
    ]
    logger.info(f"Predict called for {len(items)} items")
    return results if isinstance(data, list) else results[0]


@router.get("/dataset/preview")
async def preview(n: int = 5):
    df = load_dataset()
    return [DatasetRowChurn(**row) for row in df.head(n).to_dict(orient="records")]


@router.get("/dataset/info")
async def info():
    df = load_dataset()
    return {"rows": len(df), "columns": len(df.columns), "column_names": df.columns.tolist(), "churn_distribution": df["churn"].value_counts().to_dict()}


@router.get("/dataset/split-info")
async def split_info():
    df = load_dataset()
    x_train, x_test, y_train, y_test = prepare_data(df)
    return {"train_size": len(x_train), "test_size": len(y_test), "churn_train": y_train.value_counts().to_dict(), "churn_test": y_test.value_counts().to_dict()}


@router.post("/model/train")
async def train(config: TrainingConfigChurn):
    global loaded_model
    df = load_dataset()
    X_train, X_test, y_train, y_test = prepare_data(df)
    trained_pipeline = train_churn_model(X_train, y_train, config.model_type, config.hyperparameters)
    y_prediction = trained_pipeline.predict(X_test)
    save_churn_model(trained_pipeline, {"accuracy": accuracy_score(y_test, y_prediction), "f1": f1_score(y_test, y_prediction), "model_type": config.model_type, "hyperparameters": config.hyperparameters})
    save_history({"timestamp": datetime.now().isoformat(), "model_type": config.model_type, "hyperparameters": config.hyperparameters, "accuracy": accuracy_score(y_test, y_prediction), "f1": f1_score(y_test, y_prediction)})
    loaded_model = load_churn_model()
    logger.info(f"Training complete: accuracy={accuracy_score(y_test, y_prediction):.3f}")
    return {"accuracy": accuracy_score(y_test, y_prediction), "f1": f1_score(y_test, y_prediction)}


@router.get("/model/status")
async def model_status():
    if loaded_model is None: return {"trained": False}
    return {"trained": True, "trained_at": loaded_model["trained_at"], "model_type": loaded_model["metrics"].get("model_type"), "hyperparameters": loaded_model["metrics"].get("hyperparameters"), "metrics": {"accuracy": loaded_model["metrics"]["accuracy"], "f1": loaded_model["metrics"]["f1"]}}


@router.get("/model/schema")
async def model_schema():
    int_cols = ["support_requests", "account_age_months", "failed_payments", "autopay_enabled"]
    return {
        **{col: ("int" if col in int_cols else "float") for col in NUMERIC_COLS},
        **{col: "str" for col in CATEGORICAL_COLS}
    }


@router.get("/model/metrics")
async def model_metrics(n: int = 5): return load_history()[-n:]


@router.get("/health")
async def health():
    try:
        load_dataset()
        dataset_available = True
    except:
        dataset_available = False
    return {"status": "ok", "model_loaded": loaded_model is not None, "dataset_available": dataset_available}