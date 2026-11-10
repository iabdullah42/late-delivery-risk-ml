"""Features, train/test split, model pipelines and scoring, shared by notebooks, tests and the API."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["shipping_mode", "market", "region", "country", "segment",
               "payment_type", "main_department"]
NUMERIC = ["scheduled_days", "n_items", "n_departments", "total_qty", "order_value",
           "avg_discount_rate", "order_month", "order_dow", "order_hour"]
FEATURES = CATEGORICAL + NUMERIC


def time_split(orders: pd.DataFrame, test_frac: float = 0.2):
    """Train on the oldest orders, test on the newest, the way the model will really be used."""
    orders = orders.sort_values("order_date")
    cut = int(len(orders) * (1 - test_frac))
    return orders.iloc[:cut], orders.iloc[cut:]


def make_preprocessor(numeric=NUMERIC) -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False), CATEGORICAL),
        ("num", StandardScaler(), list(numeric)),
    ])


def make_logistic(numeric=NUMERIC) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(numeric)),
                     ("model", LogisticRegression(max_iter=2000))])


def make_forest() -> Pipeline:
    return Pipeline([("prep", make_preprocessor()),
                     ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                                      n_jobs=-1, random_state=42))])


def make_boosting() -> Pipeline:
    return Pipeline([("prep", make_preprocessor()),
                     ("model", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                              random_state=42))])


def score(name: str, y_true, prob, threshold: float = 0.5) -> dict:
    return {"model": name,
            "accuracy": round(accuracy_score(y_true, prob >= threshold), 4),
            "roc_auc": round(roc_auc_score(y_true, prob), 4)}