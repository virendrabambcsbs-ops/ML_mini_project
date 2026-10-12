"""
Unit tests for association mining, benchmarking, and recommendation engine (src/association_analysis.py, src/recommendations.py).
"""

import pytest
import pandas as pd
import numpy as np
from src.association_analysis import (
    AssociationMiner,
    benchmark_apriori_vs_fpgrowth,
    run_parameter_tuning_grid,
)
from src.recommendations import ComboRecommendationEngine


@pytest.fixture
def fast_food_one_hot():
    # 5 transactions with clear Burger + Fries co-occurrence
    data = [
        {"Burger": True, "Fries": True, "Soda": True, "Salad": False},
        {"Burger": True, "Fries": True, "Soda": False, "Salad": False},
        {"Burger": True, "Fries": True, "Soda": True, "Salad": False},
        {"Burger": False, "Fries": False, "Soda": True, "Salad": True},
        {"Burger": True, "Fries": False, "Soda": True, "Salad": False},
    ]
    return pd.DataFrame(data)


def test_fpgrowth_and_apriori_mining(fast_food_one_hot):
    miner = AssociationMiner(min_support=0.3, min_confidence=0.5, min_lift=1.0)
    
    # Test Apriori
    ap_itemsets, ap_time = miner.mine_frequent_itemsets(fast_food_one_hot, algorithm="apriori")
    assert not ap_itemsets.empty
    assert ap_time >= 0

    # Test FP-Growth
    fp_itemsets, fp_time = miner.mine_frequent_itemsets(fast_food_one_hot, algorithm="fpgrowth")
    assert not fp_itemsets.empty
    assert len(ap_itemsets) == len(fp_itemsets)

    # Test rules
    rules = miner.generate_rules(fp_itemsets, num_transactions=len(fast_food_one_hot))
    assert not rules.empty
    assert "lift" in rules.columns
    assert "confidence" in rules.columns
    assert (rules["lift"] >= 1.0).all()


def test_no_rules_scenario(fast_food_one_hot):
    # Setting an impossibly high threshold
    miner = AssociationMiner(min_support=0.99, min_confidence=0.99)
    itemsets, _ = miner.mine_frequent_itemsets(fast_food_one_hot)
    rules = miner.generate_rules(itemsets, num_transactions=len(fast_food_one_hot))
    assert rules.empty
    assert "antecedent_str" in rules.columns


def test_benchmark_apriori_vs_fpgrowth(fast_food_one_hot):
    bench = benchmark_apriori_vs_fpgrowth(
        fast_food_one_hot, min_support=0.3, min_confidence=0.4
    )
    assert "comparison_df" in bench
    assert bench["speedup"] > 0
    assert len(bench["apriori_itemsets"]) == len(bench["fpgrowth_itemsets"])


def test_parameter_tuning_grid(fast_food_one_hot):
    grid_df = run_parameter_tuning_grid(
        fast_food_one_hot,
        support_grid=[0.2, 0.4],
        confidence_grid=[0.3, 0.6],
    )
    assert len(grid_df) == 4
    assert "min_support" in grid_df.columns
    assert "frequent_itemsets" in grid_df.columns
    assert "association_rules" in grid_df.columns


def test_combo_recommendation_engine(fast_food_one_hot):
    miner = AssociationMiner(min_support=0.3, min_confidence=0.4, min_lift=1.0)
    itemsets, _ = miner.mine_frequent_itemsets(fast_food_one_hot)
    rules = miner.generate_rules(itemsets, num_transactions=len(fast_food_one_hot))

    engine = ComboRecommendationEngine(rules, top_items={"Burger": 4, "Soda": 4, "Fries": 3})

    # Recommend for basket with "Burger"
    recs = engine.recommend(["Burger"], top_k=2)
    assert len(recs) > 0
    # Should recommend Fries or Soda, but NEVER Burger itself
    rec_items = [r["item"] for r in recs]
    assert "Burger" not in rec_items

    # Fallback test for unknown item with no rules
    fallback_recs = engine.recommend(["UnknownItem"], top_k=2)
    assert len(fallback_recs) > 0
    assert fallback_recs[0]["is_fallback"] is True
