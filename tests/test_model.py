import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data import add_time_features
from src.model import FEATURES, make_boosting, make_logistic, score, time_split


def fake_orders(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "order_date": pd.date_range("2016-01-01", periods=n, freq="h"),
        "shipping_mode": rng.choice(["First Class", "Second Class", "Standard Class", "Same Day"], n),
        "market": "Europe", "region": "Western Europe",
        "country": rng.choice(["Irlanda", "Francia"], n),
        "segment": "Consumer", "payment_type": "DEBIT", "main_department": "Fan Shop",
        "n_items": rng.integers(1, 5, n), "n_departments": 1,
        "total_qty": rng.integers(1, 10, n), "order_value": rng.uniform(20, 500, n),
        "avg_discount_rate": 0.1, "scheduled_days": 4,
    })
    df["late"] = (df["shipping_mode"] == "First Class").astype(int)
    return add_time_features(df)


def test_time_split_never_trains_on_the_future():
    train, test = time_split(fake_orders())
    assert train["order_date"].max() <= test["order_date"].min()
    assert (len(train), len(test)) == (320, 80)


def test_model_learns_a_simple_rule():
    train, test = time_split(fake_orders())
    model = make_boosting().fit(train[FEATURES], train["late"])
    assert roc_auc_score(test["late"], model.predict_proba(test[FEATURES])[:, 1]) > 0.95


def test_unseen_category_does_not_crash():
    orders = fake_orders()
    model = make_logistic().fit(orders[FEATURES], orders["late"])
    new = orders.head(1).assign(country="Narnia")
    assert 0 <= model.predict_proba(new[FEATURES])[0, 1] <= 1


def test_score_returns_both_metrics():
    out = score("x", np.array([0, 1, 1]), np.array([0.1, 0.8, 0.6]))
    assert out == {"model": "x", "accuracy": 1.0, "roc_auc": 1.0}