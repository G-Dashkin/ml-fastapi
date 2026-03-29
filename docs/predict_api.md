# Описание алгоритма

## Модель
Поддерживаются две модели:
- LogisticRegression из scikit-learn с class_weight="balanced"
- RandomForestClassifier из scikit-learn с class_weight="balanced"

## Признаки
- Числовые: monthly_fee, usage_hours, support_requests, account_age_months, failed_payments, autopay_enabled
- Категориальные: region, device_type, payment_method

## Предобработка
- Числовые: StandardScaler (приводит к одному масштабу)
- Категориальные: OneHotEncoder (текст → 0 и 1)

## Pipeline
1. ColumnTransformer — разная обработка для разных колонок
2. Выбранная модель (LogisticRegression или RandomForest) — классификация churn

## Метрики
- accuracy — процент правильных ответов
- f1 — качество предсказания уходящих клиентов (редкий класс)

## Веса модели (топ признаки на LogisticRegression)
- autopay_enabled: -0.405 (автооплата = останется)
- usage_hours: -0.345 (много часов = останется)
- failed_payments: 0.281 (отказы платежей = уйдёт)
- monthly_fee: -0.001 (цена почти не влияет)

---

## Примеры запросов к /model/train

### LogisticRegression — быстрое обучение

**Запрос:**
```json
{
  "model_type": "logreg",
  "hyperparameters": {
    "max_iter": 1000,
    "C": 1.0
  }
}
```

**Ответ:**
```json
{
  "accuracy": 0.635,
  "f1": 0.411
}
```

---

### LogisticRegression — точное обучение

**Запрос:**
```json
{
  "model_type": "logreg",
  "hyperparameters": {
    "max_iter": 2000,
    "C": 0.5
  }
}
```

**Параметры:**
- `max_iter: 2000` — больше итераций для лучшей сходимости
- `C: 0.5` — сильнее штраф за ошибки (регуляризация)

**Ответ:**
```json
{
  "accuracy": 0.635,
  "f1": 0.41
}
```

---

### RandomForestClassifier — быстрое обучение

**Запрос:**
```json
{
  "model_type": "random_forest",
  "hyperparameters": {
    "n_estimators": 50,
    "max_depth": 10
  }
}
```

**Параметры:**
- `n_estimators: 50` — небольшое количество деревьев
- `max_depth: 10` — ограничиваем глубину для скорости

**Ответ:**
```json
{
  "accuracy": 0.72,
  "f1": 0.11
}
```

---

### RandomForestClassifier — точное обучение

**Запрос:**
```json
{
  "model_type": "random_forest",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 15,
    "random_state": 42
  }
}
```

**Параметры:**
- `n_estimators: 100` — оптимальное количество деревьев
- `max_depth: 15` — хорошая глубина для точности
- `random_state: 42` — фиксируем случайность для воспроизводимости

**Ответ:**
```json
{
  "accuracy": 0.7975,
  "f1": 0.129
}
```

---
### RandomForestClassifier — глубокое обучение (лучше качество)

**Запрос:**
```json
{
  "model_type": "random_forest",
  "hyperparameters": {
    "n_estimators": 200,
    "max_depth": 20,
    "random_state": 42
  }
}
```

**Параметры:**
- `n_estimators: 200` — много деревьев для лучшего усреднения
- `max_depth: 20` — глубже деревья = лучше качество (медленнее обучение)

**Ответ:**
```json
{
  "accuracy": 0.82,
  "f1": 0.15
}
```

---

## Пример запроса к /predict

### Клиент скорее всего уйдёт

**Запрос:**
```json
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
```

**Ответ:**
```json
{
  "churn": 1,
  "probability": "97.07%"
}
```

### Клиент скорее всего останется

**Запрос:**
```json
{
  "monthly_fee": 49.99,
  "usage_hours": 120,
  "support_requests": 0,
  "account_age_months": 36,
  "failed_payments": 0,
  "region": "europe",
  "device_type": "desktop",
  "payment_method": "card",
  "autopay_enabled": 1
}
```

**Ответ:**
```json
{
  "churn": 0,
  "probability": "0.33%"
}
```

**Интерпретация:**
- churn: 1 — клиент скорее всего уйдёт
- churn: 0 — клиент скорее всего останется
- probability — вероятность ухода (или остаться) в процентах
---

### POST /model/train — неверный model_type

**Запрос:**
```json
{
  "model_type": "xgboost",
  "hyperparameters": {}
}
```

**Ответ (500):**
```json
{
  "code": 500,
  "message": "Internal Server Error",
  "details": "Unknown model_type: xgboost. Use 'logreg' or 'random_forest'"
}
```