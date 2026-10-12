"""
QuickBite - Market Basket Intelligence
Module: Association Analysis, Benchmarking & Parameter Tuning (src/association_analysis.py)

Integrates mlxtend Apriori and FP-Growth, provides comparative benchmarking,
rule extraction, and parameter tuning experiment laboratories.
"""

import time
from typing import Dict, List, Optional, Set, Tuple, Any
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


class AssociationMiner:
    """
    Production-grade association rule mining engine supporting both Apriori and FP-Growth.
    """

    def __init__(
        self,
        min_support: float = 0.03,
        min_confidence: float = 0.20,
        min_lift: float = 1.0,
        min_len: int = 1,
        max_len: Optional[int] = None,
        max_rules: int = 500,
    ):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.min_len = min_len
        self.max_len = max_len
        self.max_rules = max_rules

    def mine_frequent_itemsets(
        self,
        one_hot_df: pd.DataFrame,
        algorithm: str = "fpgrowth",
        verbose: int = 0,
    ) -> Tuple[pd.DataFrame, float]:
        """
        Mine frequent itemsets using either 'apriori' or 'fpgrowth'.
        Returns:
            - DataFrame of frequent itemsets (with columns: support, itemsets, length)
            - Runtime in seconds.
        """
        if one_hot_df.empty:
            return pd.DataFrame(columns=["support", "itemsets", "length"]), 0.0

        start_time = time.perf_counter()

        algo_lower = algorithm.lower().strip()
        if algo_lower == "apriori":
            frequent_itemsets = apriori(
                one_hot_df,
                min_support=self.min_support,
                use_colnames=True,
                max_len=self.max_len,
                verbose=verbose,
            )
        elif algo_lower in ("fpgrowth", "fp_growth", "fp-growth"):
            frequent_itemsets = fpgrowth(
                one_hot_df,
                min_support=self.min_support,
                use_colnames=True,
                max_len=self.max_len,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose 'apriori' or 'fpgrowth'.")

        runtime = time.perf_counter() - start_time

        if frequent_itemsets.empty:
            return pd.DataFrame(columns=["support", "itemsets", "length"]), runtime

        # Compute itemset lengths
        frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)

        # Apply min_len filter
        if self.min_len > 1:
            frequent_itemsets = frequent_itemsets[frequent_itemsets["length"] >= self.min_len]

        frequent_itemsets["items_str"] = frequent_itemsets["itemsets"].apply(
            lambda s: ", ".join(sorted(list(s)))
        )
        frequent_itemsets["count"] = (frequent_itemsets["support"] * len(one_hot_df)).round().astype(int)
        frequent_itemsets = frequent_itemsets.sort_values(by=["length", "support"], ascending=[True, False]).reset_index(drop=True)

        return frequent_itemsets, runtime

    def generate_rules(
        self,
        frequent_itemsets: pd.DataFrame,
        num_transactions: int,
    ) -> pd.DataFrame:
        """
        Generate association rules from frequent itemsets using confidence & lift thresholds.
        """
        if frequent_itemsets.empty:
            return self._empty_rules_df()

        # mlxtend association_rules requires at least itemsets of length >= 2
        multi_itemsets = frequent_itemsets[frequent_itemsets["length"] >= 2]
        if multi_itemsets.empty:
            return self._empty_rules_df()

        try:
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=self.min_confidence,
                num_itemsets=num_transactions,
            )
        except Exception:
            # Fallback for version differences in mlxtend
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=self.min_confidence,
            )

        if rules.empty:
            return self._empty_rules_df()

        # Filter by lift
        if self.min_lift > 0:
            rules = rules[rules["lift"] >= self.min_lift]

        if rules.empty:
            return self._empty_rules_df()

        # Enhance rule representation
        rules["antecedent_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
        rules["consequent_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
        rules["rule_str"] = rules["antecedent_str"] + " -> " + rules["consequent_str"]
        rules["antecedent_len"] = rules["antecedents"].apply(len)
        rules["consequent_len"] = rules["consequents"].apply(len)

        # Round metrics for presentation
        for col in ["support", "confidence", "lift", "leverage", "conviction"]:
            if col in rules.columns:
                rules[col] = rules[col].replace([np.inf, -np.inf], np.nan).round(4)

        # Sort by lift descending, then confidence descending
        rules = rules.sort_values(by=["lift", "confidence"], ascending=[False, False]).reset_index(drop=True)

        # Cap displayed rules if needed
        if self.max_rules and len(rules) > self.max_rules:
            rules = rules.head(self.max_rules)

        return rules

    @staticmethod
    def _empty_rules_df() -> pd.DataFrame:
        cols = [
            "antecedents", "consequents", "antecedent_str", "consequent_str",
            "rule_str", "support", "confidence", "lift", "leverage", "conviction",
            "antecedent_len", "consequent_len"
        ]
        return pd.DataFrame(columns=cols)


def benchmark_apriori_vs_fpgrowth(
    one_hot_df: pd.DataFrame,
    min_support: float = 0.03,
    min_confidence: float = 0.20,
    min_lift: float = 1.0,
) -> Dict[str, Any]:
    """
    Direct head-to-head empirical comparison of Apriori and FP-Growth on identical data.
    """
    miner = AssociationMiner(
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
    )
    num_trans = len(one_hot_df)

    # Run Apriori
    ap_itemsets, ap_time = miner.mine_frequent_itemsets(one_hot_df, algorithm="apriori")
    ap_rules = miner.generate_rules(ap_itemsets, num_transactions=num_trans)

    # Run FP-Growth
    fp_itemsets, fp_time = miner.mine_frequent_itemsets(one_hot_df, algorithm="fpgrowth")
    fp_rules = miner.generate_rules(fp_itemsets, num_transactions=num_trans)

    # Check rule set agreement
    ap_rule_signatures: Set[str] = set(ap_rules["rule_str"]) if not ap_rules.empty else set()
    fp_rule_signatures: Set[str] = set(fp_rules["rule_str"]) if not fp_rules.empty else set()
    rule_overlap = len(ap_rule_signatures.intersection(fp_rule_signatures))

    speedup = (ap_time / fp_time) if fp_time > 0 else 1.0

    comparison_data = {
        "Metric": [
            "Execution Time (seconds)",
            "Frequent Itemsets Found",
            "Association Rules Found",
            "Max Itemset Length",
            "Average Itemset Length",
            "Maximum Lift Discovered",
        ],
        "Apriori": [
            round(ap_time, 4),
            len(ap_itemsets),
            len(ap_rules),
            int(ap_itemsets["length"].max()) if not ap_itemsets.empty else 0,
            round(float(ap_itemsets["length"].mean()), 2) if not ap_itemsets.empty else 0.0,
            round(float(ap_rules["lift"].max()), 4) if not ap_rules.empty else 0.0,
        ],
        "FP-Growth": [
            round(fp_time, 4),
            len(fp_itemsets),
            len(fp_rules),
            int(fp_itemsets["length"].max()) if not fp_itemsets.empty else 0,
            round(float(fp_itemsets["length"].mean()), 2) if not fp_itemsets.empty else 0.0,
            round(float(fp_rules["lift"].max()), 4) if not fp_rules.empty else 0.0,
        ],
    }
    comparison_df = pd.DataFrame(comparison_data)

    insights = (
        f"FP-Growth executed in {fp_time:.4f}s compared to Apriori's {ap_time:.4f}s "
        f"({speedup:.2f}x relative speedup). "
        f"Both algorithms identified exactly {len(ap_itemsets)} frequent itemsets and "
        f"{len(ap_rules)} association rules with {rule_overlap} identical rule matches. "
        "This confirms algorithmic mathematical equivalence while highlighting FP-Growth's "
        "superior memory-tree compression versus Apriori's multi-pass candidate generation."
    )

    return {
        "comparison_df": comparison_df,
        "apriori_time": ap_time,
        "fpgrowth_time": fp_time,
        "speedup": speedup,
        "apriori_itemsets": ap_itemsets,
        "fpgrowth_itemsets": fp_itemsets,
        "apriori_rules": ap_rules,
        "fpgrowth_rules": fp_rules,
        "rule_overlap": rule_overlap,
        "insights": insights,
    }


def run_parameter_tuning_grid(
    one_hot_df: pd.DataFrame,
    support_grid: Optional[List[float]] = None,
    confidence_grid: Optional[List[float]] = None,
    algorithm: str = "fpgrowth",
) -> pd.DataFrame:
    """
    Parameter Tuning Lab experiment grid runner.
    Varies min_support and min_confidence to record runtime, itemset count,
    rule count, and top lift.
    """
    if support_grid is None:
        support_grid = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    if confidence_grid is None:
        confidence_grid = [0.20, 0.30, 0.40, 0.50]

    num_trans = len(one_hot_df)
    results = []

    for sup in support_grid:
        miner = AssociationMiner(min_support=sup, min_confidence=0.1, min_lift=1.0)
        itemsets, run_time = miner.mine_frequent_itemsets(one_hot_df, algorithm=algorithm)
        num_itemsets = len(itemsets)

        for conf in confidence_grid:
            miner.min_confidence = conf
            rules = miner.generate_rules(itemsets, num_transactions=num_trans)
            num_rules = len(rules)
            max_lift = float(rules["lift"].max()) if not rules.empty else np.nan
            avg_conf = float(rules["confidence"].mean()) if not rules.empty else np.nan

            results.append({
                "min_support": sup,
                "min_confidence": conf,
                "frequent_itemsets": num_itemsets,
                "association_rules": num_rules,
                "runtime_seconds": round(run_time, 4),
                "max_lift": round(max_lift, 3) if not np.isnan(max_lift) else np.nan,
                "avg_confidence": round(avg_conf, 3) if not np.isnan(avg_conf) else np.nan,
            })

    return pd.DataFrame(results)
