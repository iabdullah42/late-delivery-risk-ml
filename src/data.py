"""Turn the raw DataCo file (one row per item) into one clean row per order.

Only information known when the order is placed is kept, so the model can't cheat.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv"
ORDERS_PATH = ROOT / "data" / "processed" / "orders.csv"
TARGET = "late"

# The promise for each shipping mode (fixed in this data: see notebook 04, Cell 6)
PROMISED_DAYS = {
    "First Class": 1,
    "Same Day": 0,
    "Second Class": 2,
    "Standard Class": 4,
}


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["order_month"] = df["order_date"].dt.month
    df["order_dow"] = df["order_date"].dt.day_of_week  # 0 = Monday
    df["order_hour"] = df["order_date"].dt.hour
    return df


# Known only AFTER the order ships or arrives: never use these as features.
LEAKY_COLUMNS = [
    "Delivery Status",
    "Days for shipping (real)",
    "shipping date (DateOrders)",
    "Order Status",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin-1")


def build_orders(raw: pd.DataFrame) -> pd.DataFrame:
    # Canceled and suspected-fraud orders never shipped, so "late or not" doesn't apply.
    shipped = raw[raw["Delivery Status"] != "Shipping canceled"].copy()
    shipped["order_date"] = pd.to_datetime(shipped["order date (DateOrders)"])

    orders = (
        shipped.groupby("Order Id")
        .agg(
            order_date=("order_date", "first"),
            shipping_mode=("Shipping Mode", "first"),
            scheduled_days=("Days for shipment (scheduled)", "first"),
            market=("Market", "first"),
            region=("Order Region", "first"),
            country=("Order Country", "first"),
            segment=("Customer Segment", "first"),
            payment_type=("Type", "first"),
            main_department=("Department Name", "first"),
            n_items=("Order Item Id", "size"),
            n_departments=("Department Name", "nunique"),
            total_qty=("Order Item Quantity", "sum"),
            order_value=("Order Item Total", "sum"),
            avg_discount_rate=("Order Item Discount Rate", "mean"),
            late=("Late_delivery_risk", "first"),
        )
        .reset_index()
    )

    orders = add_time_features(orders)
    return orders.sort_values("order_date").reset_index(drop=True)
