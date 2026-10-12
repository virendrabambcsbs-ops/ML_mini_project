"""
Unit tests for the first-principles Manual Apriori implementation (src/apriori_manual.py).
"""

import pytest
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori as mlx_apriori
from src.apriori_manual import ManualApriori, explain_manual_worked_example


@pytest.fixture
def textbook_transactions():
    """
    Classic textbook dataset:
    T1: {A, B, E}
    T2: {B, D}
    T3: {B, C}
    T4: {A, B, D}
    T5: {A, C}
    T6: {B, C}
    T7: {A, C}
    T8: {A, B, C, E}
    T9: {A, B, C}
    """
    return [
        ["A", "B", "E"],
        ["B", "D"],
        ["B", "C"],
        ["A", "B", "D"],
        ["A", "C"],
        ["B", "C"],
        ["A", "C"],
        ["A", "B", "C", "E"],
        ["A", "B", "C"],
    ]


def test_parameter_boundaries():
    with pytest.raises(ValueError, match="min_support must be in"):
        ManualApriori(min_support=-0.1)
    with pytest.raises(ValueError, match="min_support must be in"):
        ManualApriori(min_support=1.5)
    with pytest.raises(ValueError, match="min_confidence must be in"):
        ManualApriori(min_confidence=-0.05)
    with pytest.raises(ValueError, match="min_lift must be"):
        ManualApriori(min_lift=-1.0)


def test_manual_apriori_itemset_support(textbook_transactions):
    miner = ManualApriori(min_support=2/9, min_confidence=0.1, min_lift=0.0)
    miner.fit(textbook_transactions)

    itemsets_df = miner.get_frequent_itemsets_df()
    assert not itemsets_df.empty

    support_map = {frozenset(row["items"]): row["support"] for _, row in itemsets_df.iterrows()}

    # A appears in 6 of 9 transactions = 6/9 = 0.666667
    assert frozenset(["A"]) in support_map
    assert abs(support_map[frozenset(["A"])] - (6/9)) < 1e-4

    # B appears in 7 of 9 transactions = 7/9 = 0.777778
    assert frozenset(["B"]) in support_map
    assert abs(support_map[frozenset(["B"])] - (7/9)) < 1e-4

    # {A, B} appears in T1, T4, T8, T9 = 4/9 = 0.444444
    assert frozenset(["A", "B"]) in support_map
    assert abs(support_map[frozenset(["A", "B"])] - (4/9)) < 1e-4


def test_parity_with_mlxtend(textbook_transactions):
    """
    Mandatory Rubric Check: Verify manual implementation matches mlxtend output within floating-point tolerance.
    """
    # Build one-hot encoding for mlxtend
    all_items = sorted(list(set(item for basket in textbook_transactions for item in basket)))
    records = []
    for basket in textbook_transactions:
        records.append({item: (item in basket) for item in all_items})
    one_hot_df = pd.DataFrame(records)

    min_sup = 0.25
    # mlxtend run
    mlx_res = mlx_apriori(one_hot_df, min_support=min_sup, use_colnames=True)
    mlx_map = {frozenset(row["itemsets"]): row["support"] for _, row in mlx_res.iterrows()}

    # Manual run
    manual_miner = ManualApriori(min_support=min_sup)
    manual_miner.fit(textbook_transactions)
    manual_df = manual_miner.get_frequent_itemsets_df()
    manual_map = {frozenset(row["items"]): row["support"] for _, row in manual_df.iterrows()}

    # Check identical itemset discovery
    assert set(mlx_map.keys()) == set(manual_map.keys()), "Manual and mlxtend found different itemsets"

    # Check support equality within 1e-5 tolerance
    for itemset in mlx_map:
        assert abs(mlx_map[itemset] - manual_map[itemset]) < 1e-5, f"Support mismatch for {itemset}"


def test_apriori_subset_pruning():
    """
    Verify candidate generation prunes candidates with infrequent subsets.
    """
    miner = ManualApriori(min_support=0.5)
    # L1 contains only A, B, C
    l1 = [frozenset(["A"]), frozenset(["B"]), frozenset(["C"])]
    # C2 generation
    c2 = miner._generate_ck(l1, 2)
    assert c2 == {frozenset(["A", "B"]), frozenset(["A", "C"]), frozenset(["B", "C"])}

    # Suppose L2 has only {A, B} and {A, C} (but NOT {B, C})
    l2 = [frozenset(["A", "B"]), frozenset(["A", "C"])]
    c3 = miner._generate_ck(l2, 3)
    # Union of {A,B} and {A,C} is {A,B,C}. But {B,C} is NOT in L2, so {A,B,C} must be PRUNED!
    assert len(c3) == 0, "Candidate {A, B, C} should have been pruned because {B, C} was not in L2"


def test_explain_manual_worked_example():
    ex = explain_manual_worked_example()
    assert ex["N"] == 100
    assert ex["support_AB"] == 0.20
    assert ex["confidence_A_to_B"] == 0.50
    assert abs(ex["lift_A_to_B"] - (0.50 / 0.30)) < 0.01
    assert "Lift" in ex["interpretation"]
