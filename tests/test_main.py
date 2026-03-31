from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "ml churn service is running"}


def test_train():
    response = client.post(
        "/model/train",
        json={
            "model_type": "logreg",
            "hyperparameters": {
                "max_iter": 100
            }
        }
    )
    assert response.status_code == 200
    assert "accuracy" in response.json()
    assert "f1" in response.json()


def test_model_status():
    response = client.get("/model/status")
    assert response.status_code == 200
    assert response.json()["trained"] == True


def test_predict():
    response = client.post("/predict", json={
        "monthly_fee": 9.99,
        "usage_hours": 2,
        "support_requests": 5,
        "account_age_months": 1,
        "failed_payments": 4,
        "region": "africa",
        "device_type": "mobile",
        "payment_method": "crypto",
        "autopay_enabled": 0
    })
    assert response.status_code == 200
    assert "churn" in response.json()
    assert "probability" in response.json()


def test_predict_validation_error():
    response = client.post("/predict", json={})
    assert response.status_code == 422
    assert response.json()["code"] == 422