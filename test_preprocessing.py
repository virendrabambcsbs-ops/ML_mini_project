"""
Unit tests for data preprocessing and validation module (src/preprocessing.py).
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import TransactionPreprocessor


@pytest.fixture
def sample_transaction_df():
    data = [
        {"order_id": 101, "item_name": "Cheeseburger"},
        {"order_id": 101, "item_name": "French Fries"},
        {"order_id": 101, "item_name": "Soda"},
        {"order_id": 102, "item_name": "Cheeseburger"},
        {"order_id": 102, "item_name": "French Fries"},
        {"order_id": 103, "item_name": "Chicken Nuggets"},
        {"order_id": 103, "item_name": "Soda"},
        {"order_id": 104, "item_name": "Cheeseburger"},
        {"order_id": 104, "item_name": "Milkshake"},
    ]
    return pd.DataFrame(data)


def test_column_detection():
    df = pd.DataFrame({"Order_ID": [1], "Product_Name": ["Burger"], "Qty": [2], "Price": [5.0]})
    detected = TransactionPreprocessor.detect_columns(df)
    assert detected["order_col"] == "Order_ID"
    assert detected["item_col"] == "Product_Name"
    assert detected["quantity_col"] == "Qty"
    assert detected["price_col"] == "Price"


def test_clean_dataset_normal(sample_transaction_df):
    preprocessor = TransactionPreprocessor(order_col="order_id", item_col="item_name")
    cleaned = preprocessor.clean_dataset(sample_transaction_df)
    assert len(cleaned) == 9
    assert preprocessor.cleaning_report["final_unique_orders"] == 4
    assert preprocessor.cleaning_report["final_unique_items"] == 5


def test_clean_dataset_removes_invalid_items():
    df = pd.DataFrame([
        {"order_id": 1, "item_name": "Burger"},
        {"order_id": 1, "item_name": None},
        {"order_id": 1, "item_name": "   "},
        {"order_id": 1, "item_name": "NONE"},
        {"order_id": 2, "item_name": "Fries"},
    ])
    preprocessor = TransactionPreprocessor()
    cleaned = preprocessor.clean_dataset(df)
    assert len(cleaned) == 2
    assert set(cleaned["item_name"]) == {"Burger", "Fries"}


def test_clean_dataset_empty_raises_error():
    preprocessor = TransactionPreprocessor()
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        preprocessor.clean_dataset(pd.DataFrame())


def test_clean_dataset_missing_column_raises_error():
    df = pd.DataFrame({"bad_col": [1, 2]})
    preprocessor = TransactionPreprocessor(order_col="order_id", item_col="item_name")
    with pytest.raises(ValueError, match="Specified order column"):
        preprocessor.clean_dataset(df)


def test_create_transaction_baskets(sample_transaction_df):
    preprocessor = TransactionPreprocessor(order_col="order_id", item_col="item_name")
    trans_list, one_hot_df = preprocessor.create_transaction_baskets(sample_transaction_df)

    assert len(trans_list) == 4
    assert ["Cheeseburger", "French Fries", "Soda"] == trans_list[0]
    assert one_hot_df.shape == (4, 5)
    assert bool(one_hot_df.loc[101, "Cheeseburger"]) is True
    assert bool(one_hot_df.loc[103, "Cheeseburger"]) is False


def test_basket_statistics(sample_transaction_df):
    preprocessor = TransactionPreprocessor(order_col="order_id", item_col="item_name")
    trans_list, _ = preprocessor.create_transaction_baskets(sample_transaction_df)
    stats = preprocessor.get_basket_statistics(sample_transaction_df, trans_list)

    assert stats["total_transactions"] == 4
    assert stats["total_items"] == 9
    assert stats["unique_items"] == 5
    assert stats["avg_basket_size"] == 2.25
    assert stats["top_items"]["Cheeseburger"] == 3
