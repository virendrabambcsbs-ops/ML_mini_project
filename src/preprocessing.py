"""
QuickBite - Market Basket Intelligence
Module: Data Preprocessing and Validation (src/preprocessing.py)

Handles ingestion, quality diagnostics, cleaning, normalization,
and one-hot encoding for fast-food transaction datasets.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np


class TransactionPreprocessor:
    """
    Robust preprocessor for fast-food transaction data.
    Supports flexible column mapping, quality checks, deduplication,
    normalization, and basket transformation.
    """

    def __init__(
        self,
        order_col: str = "order_id",
        item_col: str = "item_name",
        quantity_col: Optional[str] = None,
        price_col: Optional[str] = None,
        date_col: Optional[str] = None,
    ):
        self.order_col = order_col
        self.item_col = item_col
        self.quantity_col = quantity_col
        self.price_col = price_col
        self.date_col = date_col
        self.cleaning_report: Dict[str, Any] = {}

    @staticmethod
    def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """
        Heuristically suggest order and item columns based on common names.
        """
        cols_lower = {col.lower().strip(): col for col in df.columns}
        
        # Order candidates
        order_candidates = [
            "order_id", "order id", "orderid", "order", "invoice_no",
            "invoiceno", "transaction_id", "transaction id", "bill_id",
            "receipt_no", "ticket_id"
        ]
        detected_order = None
        for cand in order_candidates:
            if cand in cols_lower:
                detected_order = cols_lower[cand]
                break

        # Item candidates
        item_candidates = [
            "item_name", "item name", "item", "product_name", "product name",
            "product", "items", "food_item", "description", "menu_item"
        ]
        detected_item = None
        for cand in item_candidates:
            if cand in cols_lower:
                detected_item = cols_lower[cand]
                break

        # Quantity candidates
        qty_candidates = ["quantity", "qty", "count", "items_count"]
        detected_qty = None
        for cand in qty_candidates:
            if cand in cols_lower:
                detected_qty = cols_lower[cand]
                break

        # Price candidates
        price_candidates = ["price", "item_price", "amount", "total", "unit_price"]
        detected_price = None
        for cand in price_candidates:
            if cand in cols_lower:
                detected_price = cols_lower[cand]
                break

        # Date candidates
        date_candidates = ["date", "order_date", "timestamp", "time", "datetime"]
        detected_date = None
        for cand in date_candidates:
            if cand in cols_lower:
                detected_date = cols_lower[cand]
                break

        # Fallback to first two columns if not detected
        columns = list(df.columns)
        if not detected_order and len(columns) > 0:
            detected_order = columns[0]
        if not detected_item and len(columns) > 1:
            detected_item = columns[1]

        return {
            "order_col": detected_order,
            "item_col": detected_item,
            "quantity_col": detected_qty,
            "price_col": detected_price,
            "date_col": detected_date,
        }

    def inspect_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive diagnostic metrics for raw data inspection.
        """
        total_rows = len(df)
        if total_rows == 0:
            return {
                "total_rows": 0,
                "total_orders": 0,
                "unique_items": 0,
                "missing_values": {},
                "duplicate_rows": 0,
                "empty_items_count": 0,
                "null_orders_count": 0,
                "columns": list(df.columns),
                "dtypes": {},
            }

        missing_summary = df.isnull().sum().to_dict()
        duplicate_rows = int(df.duplicated().sum())

        order_col = self.order_col if self.order_col in df.columns else None
        item_col = self.item_col if self.item_col in df.columns else None

        total_orders = int(df[order_col].dropna().nunique()) if order_col else 0
        null_orders_count = int(df[order_col].isnull().sum()) if order_col else 0

        empty_items_count = 0
        unique_items = 0
        if item_col:
            empty_items_mask = (
                df[item_col].isnull()
                | (df[item_col].astype(str).str.strip() == "")
                | (df[item_col].astype(str).str.lower().isin(["none", "nan", "null"]))
            )
            empty_items_count = int(empty_items_mask.sum())
            unique_items = int(df[~empty_items_mask][item_col].astype(str).str.strip().nunique())

        return {
            "total_rows": total_rows,
            "total_orders": total_orders,
            "unique_items": unique_items,
            "missing_values": missing_summary,
            "duplicate_rows": duplicate_rows,
            "empty_items_count": empty_items_count,
            "null_orders_count": null_orders_count,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }

    def clean_dataset(
        self,
        df: pd.DataFrame,
        drop_missing: bool = True,
        case_normalization: str = "title",  # 'title', 'upper', 'lower', or 'none'
        strip_whitespace: bool = True,
        min_item_frequency: int = 1,
        min_basket_size: int = 1,
    ) -> pd.DataFrame:
        """
        Clean and normalize raw fast-food transaction rows.
        Does NOT silently discard line items. Retains explainable cleaning report.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        if self.order_col not in df.columns:
            raise ValueError(f"Specified order column '{self.order_col}' not found in DataFrame.")
        if self.item_col not in df.columns:
            raise ValueError(f"Specified item column '{self.item_col}' not found in DataFrame.")

        initial_rows = len(df)
        cleaned_df = df.copy()

        # Step 1: Remove invalid order identifiers
        cleaned_df = cleaned_df[cleaned_df[self.order_col].notnull()]
        rows_after_order_filter = len(cleaned_df)

        # Step 2: Handle missing, empty, or placeholder item names
        if strip_whitespace:
            cleaned_df[self.item_col] = cleaned_df[self.item_col].astype(str).str.strip()

        invalid_item_placeholders = {"nan", "none", "null", "n/a", "", "undefined"}
        item_is_invalid = (
            cleaned_df[self.item_col].isnull()
            | cleaned_df[self.item_col].astype(str).str.lower().isin(invalid_item_placeholders)
        )
        if drop_missing:
            cleaned_df = cleaned_df[~item_is_invalid]
        rows_after_item_filter = len(cleaned_df)

        # Step 3: Text normalization (consistent menu item names)
        if case_normalization == "title":
            cleaned_df[self.item_col] = cleaned_df[self.item_col].str.title()
        elif case_normalization == "upper":
            cleaned_df[self.item_col] = cleaned_df[self.item_col].str.upper()
        elif case_normalization == "lower":
            cleaned_df[self.item_col] = cleaned_df[self.item_col].str.lower()

        # Step 4: Frequency filtering (rare noise filtering)
        if min_item_frequency > 1:
            item_counts = cleaned_df[self.item_col].value_counts()
            frequent_items = item_counts[item_counts >= min_item_frequency].index
            cleaned_df = cleaned_df[cleaned_df[self.item_col].isin(frequent_items)]
        rows_after_freq_filter = len(cleaned_df)

        # Step 5: Filter by minimum basket size (orders with fewer items than min_basket_size)
        order_basket_sizes = cleaned_df.groupby(self.order_col)[self.item_col].nunique()
        qualifying_orders = order_basket_sizes[order_basket_sizes >= min_basket_size].index
        cleaned_df = cleaned_df[cleaned_df[self.order_col].isin(qualifying_orders)]
        rows_after_basket_filter = len(cleaned_df)

        # Clean price if present (remove currency symbols like $)
        if self.price_col and self.price_col in cleaned_df.columns:
            if cleaned_df[self.price_col].dtype == object:
                cleaned_df[self.price_col] = (
                    cleaned_df[self.price_col]
                    .astype(str)
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                )
                cleaned_df[self.price_col] = pd.to_numeric(cleaned_df[self.price_col], errors="coerce")

        self.cleaning_report = {
            "initial_rows": initial_rows,
            "dropped_null_orders": initial_rows - rows_after_order_filter,
            "dropped_invalid_items": rows_after_order_filter - rows_after_item_filter,
            "dropped_low_frequency_items": rows_after_item_filter - rows_after_freq_filter,
            "dropped_subminimum_baskets": rows_after_freq_filter - rows_after_basket_filter,
            "final_clean_rows": rows_after_basket_filter,
            "final_unique_orders": int(cleaned_df[self.order_col].nunique()),
            "final_unique_items": int(cleaned_df[self.item_col].nunique()),
        }

        return cleaned_df

    def create_transaction_baskets(
        self, df: pd.DataFrame, binary_presence: bool = True
    ) -> Tuple[List[List[str]], pd.DataFrame]:
        """
        Construct transaction baskets from cleaned DataFrame.
        Returns:
            - transaction_list: List of item lists per order (e.g. [['Burger', 'Fries'], ...])
            - one_hot_df: Binary boolean DataFrame indexed by Order ID with columns for each item.
        """
        if df.empty:
            return [], pd.DataFrame()

        # Group items by order
        grouped = df.groupby(self.order_col)[self.item_col]
        
        if binary_presence:
            # Binary presence: duplicate items in the same basket are treated as present (True)
            transaction_list = [sorted(list(set(items))) for _, items in grouped]
            # Fast crosstab for one-hot encoding
            cross_tab = pd.crosstab(df[self.order_col], df[self.item_col])
            one_hot_df = (cross_tab > 0).astype(bool)
        else:
            # Quantity-aware or occurrence-aware grouping
            transaction_list = [list(items) for _, items in grouped]
            cross_tab = pd.crosstab(df[self.order_col], df[self.item_col])
            one_hot_df = cross_tab.astype(int)

        return transaction_list, one_hot_df

    def get_basket_statistics(
        self, df: pd.DataFrame, transaction_list: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Compute descriptive fast-food basket metrics.
        """
        if not transaction_list:
            return {
                "total_transactions": 0,
                "total_items": 0,
                "unique_items": 0,
                "avg_basket_size": 0.0,
                "median_basket_size": 0.0,
                "min_basket_size": 0,
                "max_basket_size": 0,
                "single_item_orders": 0,
                "multi_item_orders": 0,
                "top_items": {},
                "basket_size_distribution": {},
            }

        basket_sizes = [len(basket) for basket in transaction_list]
        total_transactions = len(transaction_list)

        # Item frequency counts
        all_items = [item for basket in transaction_list for item in basket]
        item_series = pd.Series(all_items)
        item_frequencies = item_series.value_counts()
        item_support = (item_frequencies / total_transactions).round(4)

        # Basket size frequency
        size_series = pd.Series(basket_sizes)
        size_distribution = size_series.value_counts().sort_index().to_dict()

        return {
            "total_transactions": total_transactions,
            "total_items": len(all_items),
            "unique_items": int(item_series.nunique()),
            "avg_basket_size": round(float(np.mean(basket_sizes)), 2),
            "median_basket_size": float(np.median(basket_sizes)),
            "min_basket_size": int(np.min(basket_sizes)),
            "max_basket_size": int(np.max(basket_sizes)),
            "single_item_orders": int((size_series == 1).sum()),
            "multi_item_orders": int((size_series > 1).sum()),
            "top_items": item_frequencies.head(15).to_dict(),
            "item_support": item_support.head(15).to_dict(),
            "basket_size_distribution": size_distribution,
        }
