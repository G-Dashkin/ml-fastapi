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

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]
)


def train_churn_model(X_train, y_train): return pipeline.fit(X_train, y_train)


def save_churn_model(trained_pipeline, metrics: dict):
    joblib.dump({
        "pipeline": trained_pipeline,
        "trained_at": datetime.datetime.now().isoformat(),
        "metrics": metrics
    }, MODEL_PATH)


def load_churn_model():
    if MODEL_PATH.exists(): return joblib.load(MODEL_PATH)
    return None