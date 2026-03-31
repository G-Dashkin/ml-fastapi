from datetime import datetime

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sklearn.metrics import accuracy_score, f1_score
from src.dataset import load_dataset
from src.models import FeatureVectorChurn, DatasetRowChurn, PredictionResponseChurn, TrainingConfigChurn, ErrorResponse
from src.model import train_churn_model, save_churn_model, load_churn_model, save_history, load_history
from src.preprocessing import prepare_data, NUMERIC_COLS, CATEGORICAL_COLS
import pandas as pd

app = FastAPI()
loaded_model = load_churn_model()


@app.get("/")
async def root(): return {"message": "ml churn service is running"}


@app.post("/predict")
async def predict(data: FeatureVectorChurn):
    # Проверяем что модель обучена
    if loaded_model is None: raise HTTPException(status_code=400, detail="Модель не обучена. Сначала переходим на /model/train")
    df = pd.DataFrame(data=[data.dict()], columns=NUMERIC_COLS+CATEGORICAL_COLS)  # Конвертируем данные клиента в DataFrame

    # Получаем предсказание и вероятность
    prediction = loaded_model["pipeline"].predict(df)[0]
    probability = loaded_model["pipeline"].predict_proba(df)[0][1]

    return PredictionResponseChurn(
        churn=int(prediction.item()),
        probability=str(round(float(probability) * 100, 2)) + "%"
    )


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
async def train(config: TrainingConfigChurn):
    df = load_dataset()                                       # Загружаем данные из дата сета
    X_train, X_test, y_train, y_test = prepare_data(df)       # Подготавливаем данные, получаем 80% на обучение и 20% тестовых

    # Обучаем модель (на 80% данных), указываем модель и передаем параметры
    trained_pipeline = train_churn_model(X_train, y_train, config.model_type, config.hyperparameters)
    y_prediction = trained_pipeline.predict(X_test)           # Тестируем модель на оставшихся 20%
    save_churn_model(trained_pipeline, {               # Сохраняем обученную модель
        "accuracy": accuracy_score(y_test, y_prediction),
        "f1": f1_score(y_test, y_prediction),
        "model_type": config.model_type,
        "hyperparameters": config.hyperparameters
    })
    save_history({
        "timestamp": datetime.now().isoformat(),
        "model_type": config.model_type,
        "hyperparameters": config.hyperparameters,
        "accuracy": accuracy_score(y_test, y_prediction),
        "f1": f1_score(y_test, y_prediction)
    })
    global loaded_model
    loaded_model = load_churn_model()
    return {
        "accuracy": accuracy_score(y_test, y_prediction),     # Выводим % правильных ответов
        "f1": f1_score(y_test, y_prediction)                  # Выводим качество предсказания редкого класса (ушедших)
    }


@app.get("/model/status")
async def model_status():
    if loaded_model is None: return {"trained": False}
    return {
        "trained": True,
        "trained_at": loaded_model["trained_at"],
        "model_type": loaded_model["metrics"].get("model_type"),
        "hyperparameters": loaded_model["metrics"].get("hyperparameters"),
        "metrics": {
            "accuracy": loaded_model["metrics"]["accuracy"],
            "f1": loaded_model["metrics"]["f1"]
        }
    }


@app.get("/model/schema")
async def model_schema():
    return {
        **{col: "float" for col in NUMERIC_COLS},
        **{col: "str" for col in CATEGORICAL_COLS}
    }


@app.get("/model/metrics")
async def model_metrics(n: int = 5): return load_history()[-n:]


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            code=exc.status_code,
            message=exc.detail,
            details=""
        ).dict()
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            code=422,
            message="Неверные данные запроса",
            details=str(exc.errors())
        ).dict()
    )