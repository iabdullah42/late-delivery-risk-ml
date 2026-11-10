"""Train the final model and save it, with its settings, for the API."""
import joblib
import pandas as pd

from src.data import ORDERS_PATH, ROOT, TARGET
from src.model import FEATURES, make_boosting, score, time_split

MODEL_PATH = ROOT / "models" / "late_risk.joblib"
THRESHOLD = 0.20   # flag when P(late) >= $3 flag cost / $15 missed-late cost (notebook 06)


def train_and_save() -> dict:
    orders = pd.read_csv(ORDERS_PATH, parse_dates=["order_date"])

    # 1. Measure it honestly on the newest 20% (the number we report)...
    train, test = time_split(orders)
    check = make_boosting().fit(train[FEATURES], train[TARGET])
    test_scores = score("gradient boosting", test[TARGET],
                        check.predict_proba(test[FEATURES])[:, 1], THRESHOLD)

    # 2. ...then retrain on ALL orders, so the deployed model has seen the most recent data.
    final = make_boosting().fit(orders[FEATURES], orders[TARGET])
    bundle = {"model": final, "features": FEATURES, "threshold": THRESHOLD,
              "trained_through": f"{orders['order_date'].max():%Y-%m-%d}",
              "test_scores": test_scores}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


if __name__ == "__main__":
    b = train_and_save()
    print("saved", MODEL_PATH.name, "| trained through", b["trained_through"], "| test", b["test_scores"])