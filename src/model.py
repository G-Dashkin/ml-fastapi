from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from src.preprocessing import NUMERIC_COLS, CATEGORICAL_COLS
import joblib
import datetime
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "churn_model.joblib"

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS)
    ]
)


def train_churn_model(X_train, y_train, model_type, hyperparameters):
    if model_type == "logreg": classifier = LogisticRegression(**hyperparameters, class_weight="balanced")
    elif model_type == "random_forest": classifier = RandomForestClassifier(**hyperparameters, class_weight="balanced")
    else: raise ValueError(f"Unknown model_type: {model_type}. Use 'logreg' or 'random_forest'")

    # Создаём pipeline с выбранной моделью
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    return pipeline.fit(X_train, y_train)


def save_churn_model(trained_pipeline, metrics: dict):
    joblib.dump({
        "pipeline": trained_pipeline,
        "trained_at": datetime.datetime.now().isoformat(),
        "metrics": metrics
    }, MODEL_PATH)


def load_churn_model():
    if MODEL_PATH.exists(): return joblib.load(MODEL_PATH)
    return None