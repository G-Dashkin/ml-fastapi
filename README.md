# ML Churn Service

FastAPI сервис для предсказания оттока клиентов.

## Запуск
pip install -r requirements.txt
uvicorn src.main:app --reload

## Эндпоинты
- GET /                  — статус сервиса
- GET /dataset/preview   — первые N строк датасета
- GET /dataset/info      — статистика датасета
- GET /dataset/split-info — размеры train/test выборок
- POST /model/train      — обучение модели
- GET /model/status      — статус и метрики модели
- POST /predict          — предсказание churn

## Структура проекта
ml-fastapi/
├── README.md
├── requirements.txt
├── data/
│   └── churn_dataset.csv
├── models/
│   └── churn_model.joblib
├── src/
│   ├── main.py            (FastAPI эндпоинты)
│   ├── models.py          (Pydantic модели данных)
│   ├── dataset.py         (загрузка датасета)
│   ├── preprocessing.py   (подготовка данных)
│   └──  model.py          (обучение и сохранение ML модели)
├── tests/
│   └── test_main.py
└── docs/
    └── algorithm.md