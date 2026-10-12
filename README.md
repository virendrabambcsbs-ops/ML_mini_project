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

Run the tests with:

```bash
pytest
```