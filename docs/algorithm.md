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