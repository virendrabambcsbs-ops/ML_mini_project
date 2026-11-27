# QuickBite ML Operations

QuickBite is a containerized Linux ML service for fast-food market basket
analysis, association-rule mining, and combo recommendations. Podman is the
recommended runtime for building, deploying, and administering the application.

## Linux and Podman operations

Verify Podman and inspect the local container environment:

```bash
podman --version
podman info
podman ps
```

Build and start the Streamlit service:

```bash
podman build -t quickbite-market-analysis -f Containerfile .
podman run --name quickbite --replace -p 8501:8501 quickbite-market-analysis
```

Monitor and administer the running workload:

```bash
podman logs -f quickbite
podman stats quickbite
podman inspect quickbite
podman restart quickbite
podman stop quickbite
podman rm quickbite
```

The application listens on TCP port `8501`. Open
`http://localhost:8501` after the container reaches the ready state.

## ML application

QuickBite provides a Streamlit interface for inspecting transactions, mining
frequent itemsets, generating association rules, and recommending combo items.
It can load the bundled dataset or accept a custom CSV export.

## Project layout

- `app.py`: Streamlit application entry point
- `src/`: preprocessing, association-rule mining, recommendation, reporting, and visualization modules
- `data/`: source datasets and dataset-generation script
- `notebooks/`: reproducible analysis notebook and notebook builder
- `tests/`: automated test suite
- `outputs/`: generated reports and exported analysis results

## Development workflow

For host-based development, create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run the test suite before rebuilding or deploying:

```bash
pytest
```

## Custom CSV format

To use **Upload Custom CSV**, provide a comma-separated file with a header row
and one row per item in an order. The file must include:

- an order/transaction identifier column, such as `order_id` or `transaction_id`
- an item/product name column, such as `item_name` or `product_name`

The same order identifier should appear on multiple rows when an order contains
multiple items. Optional columns such as `quantity`, `price`, and `date` are
supported.

```text
order_id,item_name,quantity,price
```

After uploading, select the correct **Order ID Column** and **Item / Product
Column** in the sidebar if automatic detection is incorrect.