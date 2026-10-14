# QuickBite Market Basket Analysis

QuickBite is a Streamlit application for fast-food market basket analysis,
association-rule mining, and combo recommendations.

## Project layout

- `app.py`: Streamlit application entry point
- `src/`: reusable preprocessing, mining, recommendation, reporting, and visualization code
- `data/`: source datasets and the dataset-generation script
- `notebooks/`: reproducible analysis notebook and notebook builder
- `tests/`: automated tests
- `outputs/`: generated reports and exported analysis results

## Run the application

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Custom CSV format

To use **Upload Custom CSV**, provide a comma-separated file with a header row
and one row per item in an order. The file must include:

- an order/transaction identifier column, such as `order_id` or `transaction_id`;
- an item/product name column, such as `item_name` or `product_name`.

The same order identifier should appear on multiple rows when the order
contains multiple items. Optional columns such as `quantity`, `price`, and
`date` are allowed but are not required.

The template contains only the column headers:

```text
order_id,item_name,quantity,price
```

After uploading, select the correct **Order ID Column** and **Item / Product
Column** in the sidebar if the automatic detection does not choose them.

Run the tests with:

```bash
pytest
```