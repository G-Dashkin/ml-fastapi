from sklearn.model_selection import train_test_split

NUMERIC_COLS = ["monthly_fee", "usage_hours", "support_requests", "account_age_months", "failed_payments", "autopay_enabled"]
CATEGORICAL_COLS = ["region", "device_type", "payment_method"]


def prepare_data(df):
    # заполняем пропуски
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0)
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna("unknown")

    # разделяем x и y
    x = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df["churn"]

    # делим на train и test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test