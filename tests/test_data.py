import pandas as pd

from src.data import LEAKY_COLUMNS, build_orders


def fake_raw():
    base = {"Market": "Europe", "Order Region": "Western Europe", "Order Country": "Irlanda",
            "Customer Segment": "Consumer", "Type": "DEBIT", "Department Name": "Fan Shop",
            "Order Item Quantity": 2, "Order Item Total": 100.0, "Order Item Discount Rate": 0.1,
            "Days for shipping (real)": 3, "Order Status": "COMPLETE"}
    cols = ["Order Id", "Order Item Id", "Delivery Status", "order date (DateOrders)",
            "Shipping Mode", "Days for shipment (scheduled)", "Late_delivery_risk"]
    rows = [
        (1, 11, "Late delivery",     "1/5/2016 14:10", "Second Class",   2, 1),
        (1, 12, "Late delivery",     "1/5/2016 14:10", "Second Class",   2, 1),
        (2, 21, "Shipping on time",  "1/6/2016 09:00", "Standard Class", 4, 0),
        (3, 31, "Shipping canceled", "1/7/2016 10:00", "Standard Class", 4, 0),
    ]
    return pd.DataFrame([{**base, **dict(zip(cols, r))} for r in rows])


def test_canceled_orders_are_removed():
    assert 3 not in build_orders(fake_raw())["Order Id"].values


def test_one_row_per_order_with_basket_totals():
    orders = build_orders(fake_raw()).set_index("Order Id")
    assert len(orders) == 2
    assert orders.loc[1, "n_items"] == 2
    assert orders.loc[1, "total_qty"] == 4
    assert orders.loc[1, "order_value"] == 200.0


def test_no_after_the_fact_columns():
    orders = build_orders(fake_raw())
    assert not set(LEAKY_COLUMNS) & set(orders.columns)
    assert "real_days" not in orders.columns


def test_time_features():
    order = build_orders(fake_raw()).set_index("Order Id").loc[1]
    assert order["order_hour"] == 14
    assert order["order_dow"] == 1          # Jan 5, 2016 was a Tuesday