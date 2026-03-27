# Описание алгоритма

## Модель
LogisticRegression из scikit-learn

## Признаки
- Числовые: monthly_fee, usage_hours, support_requests, account_age_months, failed_payments, autopay_enabled
- Категориальные: region, device_type, payment_method

## Предобработка
- Числовые: StandardScaler
- Категориальные: OneHotEncoder