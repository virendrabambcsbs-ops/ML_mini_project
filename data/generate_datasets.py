"""
QuickBite Dataset Generator & Fetcher
Generates:
1. `data/quickbite_synthetic_orders.csv`: 5,000 realistic fast-food restaurant orders.
2. `data/chipotle_orders.csv`: Real-world fast-food transactions from Chipotle.
"""

import os
import random
import datetime
import requests
import pandas as pd
import numpy as np

def generate_synthetic_fastfood_data(output_path: str, num_orders: int = 5000, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    menu_catalog = {
        "Classic Cheeseburger": {"category": "Burgers", "price": 5.99},
        "Double Bacon Burger": {"category": "Burgers", "price": 7.49},
        "Crispy Chicken Burger": {"category": "Burgers", "price": 6.29},
        "Spicy Veggie Burger": {"category": "Burgers", "price": 5.79},
        "Mushroom Swiss Burger": {"category": "Burgers", "price": 6.99},
        "Large French Fries": {"category": "Sides", "price": 3.49},
        "Medium French Fries": {"category": "Sides", "price": 2.79},
        "Crispy Onion Rings": {"category": "Sides", "price": 3.89},
        "Loaded Cheesy Fries": {"category": "Sides", "price": 4.49},
        "Coleslaw": {"category": "Sides", "price": 2.29},
        "Fountain Soda (Cola)": {"category": "Beverages", "price": 2.19},
        "Diet Cola": {"category": "Beverages", "price": 2.19},
        "Lemon Lime Soda": {"category": "Beverages", "price": 2.19},
        "Iced Tea": {"category": "Beverages", "price": 2.39},
        "Vanilla Milkshake": {"category": "Beverages", "price": 3.99},
        "Chocolate Milkshake": {"category": "Beverages", "price": 3.99},
        "Bottled Water": {"category": "Beverages", "price": 1.79},
        "6pc Chicken Nuggets": {"category": "Chicken & Snacks", "price": 4.59},
        "10pc Chicken Nuggets": {"category": "Chicken & Snacks", "price": 6.89},
        "Crispy Chicken Tenders": {"category": "Chicken & Snacks", "price": 6.49},
        "BBQ Dipping Sauce": {"category": "Sauces", "price": 0.50},
        "Honey Mustard Sauce": {"category": "Sauces", "price": 0.50},
        "Grilled Chicken Wrap": {"category": "Wraps", "price": 5.99},
        "Spicy Ranch Wrap": {"category": "Wraps", "price": 6.19},
        "Hot Apple Pie": {"category": "Desserts", "price": 1.99},
        "Chocolate Sundae": {"category": "Desserts", "price": 2.49},
        "Chocolate Chip Cookie": {"category": "Desserts", "price": 1.49},
    }

    # Core combination archetypes reflecting authentic dining habits
    combos = [
        # Classic Burger Combo
        (["Classic Cheeseburger", "Large French Fries", "Fountain Soda (Cola)"], 0.28),
        # Bacon Burger Combo
        (["Double Bacon Burger", "Large French Fries", "Fountain Soda (Cola)"], 0.18),
        # Chicken Burger + Shake
        (["Crispy Chicken Burger", "Medium French Fries", "Vanilla Milkshake"], 0.16),
        # Nuggets Combo
        (["10pc Chicken Nuggets", "Large French Fries", "BBQ Dipping Sauce", "Fountain Soda (Cola)"], 0.15),
        # Tenders + Sauce + Drink
        (["Crispy Chicken Tenders", "Honey Mustard Sauce", "Medium French Fries", "Iced Tea"], 0.12),
        # Veggie Combo
        (["Spicy Veggie Burger", "Crispy Onion Rings", "Lemon Lime Soda"], 0.10),
        # Wrap + Water combo
        (["Grilled Chicken Wrap", "Bottled Water"], 0.10),
        # Late Night Treat / Dessert
        (["Vanilla Milkshake", "Hot Apple Pie"], 0.08),
        (["Double Bacon Burger", "Loaded Cheesy Fries", "Chocolate Milkshake"], 0.09),
    ]

    items_list = list(menu_catalog.keys())
    start_date = datetime.datetime(2025, 1, 1, 10, 0, 0)

    rows = []
    for order_num in range(1, num_orders + 1):
        order_id = 10000 + order_num
        time_offset = datetime.timedelta(
            days=random.randint(0, 90),
            hours=random.randint(0, 12),
            minutes=random.randint(0, 59),
        )
        order_time = (start_date + time_offset).strftime("%Y-%m-%d %H:%M:%S")

        # Determine items for this basket
        basket_items = []

        # Check if matches one of the signature combos
        for combo_items, prob in combos:
            if random.random() < prob:
                basket_items.extend(combo_items)
                break

        # If no combo triggered or with random probability, pick items independently
        if not basket_items or random.random() < 0.35:
            num_additional = random.choices([1, 2, 3, 4], weights=[0.4, 0.35, 0.18, 0.07])[0]
            picked = random.sample(items_list, k=num_additional)
            basket_items.extend(picked)

        # Remove duplicate items within the basket to simulate distinct order items
        basket_items = list(dict.fromkeys(basket_items))

        # Add items to row records
        for item in basket_items:
            meta = menu_catalog[item]
            qty = random.choices([1, 2], weights=[0.88, 0.12])[0]
            rows.append({
                "order_id": order_id,
                "order_date": order_time,
                "item_name": item,
                "category": meta["category"],
                "quantity": qty,
                "price": meta["price"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic fast-food dataset: {len(df)} rows across {df['order_id'].nunique()} orders at {output_path}")
    return df

def fetch_chipotle_real_dataset(output_path: str):
    url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        import io
        tsv_data = io.StringIO(resp.text)
        df = pd.read_csv(tsv_data, sep="\t")
        df.to_csv(output_path, index=False)
        print(f"Downloaded real Chipotle dataset: {len(df)} rows across {df['order_id'].nunique()} orders at {output_path}")
        return df
    except Exception as e:
        print(f"Warning: Could not download Chipotle dataset ({e}). Fallback to synthetic.")
        return None

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir)
    
    synth_path = os.path.join(data_dir, "quickbite_synthetic_orders.csv")
    generate_synthetic_fastfood_data(synth_path, num_orders=5000)

    chipotle_path = os.path.join(data_dir, "chipotle_orders.csv")
    fetch_chipotle_real_dataset(chipotle_path)
