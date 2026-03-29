# Описание алгоритма

## Модель
LogisticRegression из scikit-learn с class_weight="balanced"

## Признаки
- Числовые: monthly_fee, usage_hours, support_requests, account_age_months, failed_payments, autopay_enabled
- Категориальные: region, device_type, payment_method

## Предобработка
- Числовые: StandardScaler (приводит к одному масштабу)
- Категориальные: OneHotEncoder (текст → 0 и 1)

## Pipeline
1. ColumnTransformer — разная обработка для разных колонок
2. LogisticRegression — классификация churn

## Метрики
- accuracy — процент правильных ответов
- f1 — качество предсказания уходящих клиентов (редкий класс)

## Веса модели (топ признаки)
- autopay_enabled: -0.405 (автооплата = останется)
- usage_hours: -0.345 (много часов = останется)
- failed_payments: 0.281 (отказы платежей = уйдёт)
- monthly_fee: -0.001 (цена почти не влияет)

## Пример запроса к /predict

**Запрос:**
POST /predict
Content-Type: application/json

{
  "monthly_fee": 9.99,
  "usage_hours": 2,
  "support_requests": 5,
  "account_age_months": 1,
  "failed_payments": 4,
  "region": "africa",
  "device_type": "mobile",
  "payment_method": "crypto",
  "autopay_enabled": 0
}

**Ответ:**
{
  "churn": 1,
  "probability": "97.07%"
}

**Интерпретация:**
- churn: 1 — клиент скорее всего уйдёт
- churn: 0 — клиент скорее всего останется
- probability — вероятность ухода в процентах