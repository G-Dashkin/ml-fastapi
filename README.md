# ML Churn Service

FastAPI сервис для предсказания оттока клиентов.

## Запуск
```
pip install -r requirements.txt
uvicorn src.main:app --reload
```

## Эндпоинты
- GET / — статус сервиса
- GET /dataset/preview — первые N строк датасета
- GET /dataset/info — статистика датасета
- GET /dataset/split-info — размеры train/test выборок
- POST /model/train — обучение модели
- POST /predict — предсказание churn

## Структура проекта
```
ml-fastapi/
├── README.md
├── requirements.txt
├── data/
│   └── churn_dataset.csv
├── src/
│   ├── main.py            (основной файл FastAPI)
│   ├── models.py          (Pydantic модели данных)
│   ├── dataset.py         (загрузка датасета)
│   ├── preprocessing.py   (подготовка данных)
│   └── model.py           (обучение ML модели)
├── tests/
│   └── test_main.py       (тесты)
└── docs/
    └── algorithm.md       (описание алгоритма)
```