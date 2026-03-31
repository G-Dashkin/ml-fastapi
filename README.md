# ML Churn Service

FastAPI сервис для предсказания оттока клиентов.

## Запуск

```bash
pip install -r requirements.txt
uvicorn src.main:app --reload
```

Swagger UI: http://127.0.0.1:8000/docs

## Запуск в Docker

```bash
docker build -t ml-fastapi .
docker run -p 8000:8000 ml-fastapi
```

## Тесты

```bash
python -m pytest tests/test_main.py -v
```

---

## Эндпоинты

| Метод | URL | Описание |
|---|---|---|
| GET | / | Статус сервиса |
| GET | /health | Состояние сервиса и модели |
| GET | /dataset/preview | Первые N строк датасета |
| GET | /dataset/info | Статистика датасета |
| GET | /dataset/split-info | Размеры train/test выборок |
| POST | /model/train | Обучение модели |
| GET | /model/status | Статус и метрики модели |
| GET | /model/schema | Список признаков и типов |
| GET | /model/metrics | История обучений |
| POST | /predict | Предсказание churn |

---

## Структура проекта

```
ml-fastapi/
├── Dockerfile
├── requirements.txt
├── data/
│   └── churn_dataset.csv
├── models/
│   ├── churn_model.joblib
│   └── training_history.json
├── src/
│   ├── main.py              — FastAPI app, обработчики ошибок
│   ├── api/
│   │   └── endpoints.py     — все эндпоинты
│   ├── ml/
│   │   ├── dataset.py       — загрузка датасета
│   │   ├── model.py         — обучение и сохранение модели
│   │   └── preprocessing.py — подготовка данных
│   └── schemas/
│       └── churn_models.py  — Pydantic модели
├── tests/
│   └── test_main.py
└── docs/
    └── predict_api.md
```

---

## Датасет

2000 строк, 10 колонок. Несбалансированный: ~80% остаются, ~20% уходят.

- Числовые: `monthly_fee`, `usage_hours`, `support_requests`, `account_age_months`, `failed_payments`, `autopay_enabled`
- Категориальные: `region`, `device_type`, `payment_method`
- Целевая: `churn` (0 — остался, 1 — ушёл)

---
## Метрики
- **Accuracy** — процент правильных предсказаний (общее качество)
- **F1** — гармоническое среднее precision и recall (качество на редком классе)
Данные несбалансированы (80% остаются, 20% уходят), поэтому F1 более важна чем accuracy.
---