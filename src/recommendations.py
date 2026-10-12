"""
QuickBite - Market Basket Intelligence
Module: Combo Recommendation Engine (src/recommendations.py)

Generates real-time, rule-backed combo meal recommendations for customer baskets,
with full explainability, confidence/lift ranking, and graceful fallback handling.
"""

from typing import Dict, List, Optional, Set, Any
import pandas as pd
import numpy as np


class ComboRecommendationEngine:
    """
    Fast-food combo recommendation system powered by association rules.
    """

    def __init__(self, rules_df: pd.DataFrame, top_items: Optional[Dict[str, int]] = None):
        """
        Args:
            rules_df: DataFrame of discovered association rules.
            top_items: Dictionary of top overall food items (for popularity fallback).
        """
        self.rules_df = rules_df
        self.top_items = top_items or {}

    def recommend(
        self,
        basket: List[str],
        top_k: int = 5,
        ranking_metric: str = "confidence",  # 'confidence', 'lift', or 'score'
        min_confidence: float = 0.10,
        min_lift: float = 1.0,
        allow_subset_antecedent: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a hypothetical customer basket.
        
        Args:
            basket: List of food items currently in customer's order.
            top_k: Maximum number of suggested items to return.
            ranking_metric: Metric to prioritize suggestions ('confidence', 'lift', 'score').
            min_confidence: Minimum rule confidence to qualify.
            min_lift: Minimum rule lift to qualify.
            allow_subset_antecedent: If True, matches any rule whose antecedent is a subset of basket.
        
        Returns:
            List of recommendation dictionaries with item name, metrics, and business rationale.
        """
        if not basket or self.rules_df.empty:
            return self._fallback_recommendations(basket, top_k)

        basket_set = set(str(item).strip() for item in basket)
        matching_rules = []

        for _, row in self.rules_df.iterrows():
            antecedent = set(row["antecedents"])
            consequent = set(row["consequents"])

            # Check rule eligibility
            if allow_subset_antecedent:
                matches_basket = antecedent.issubset(basket_set) and len(antecedent) > 0
            else:
                matches_basket = (antecedent == basket_set)

            if not matches_basket:
                continue

            # Candidate items to recommend = consequent items not already in the basket
            new_items = consequent - basket_set
            if not new_items:
                continue

            conf = float(row.get("confidence", 0.0))
            lift = float(row.get("lift", 1.0))

            if conf < min_confidence or lift < min_lift:
                continue

            # Composite hybrid score: confidence * log1p(lift)
            score = conf * np.log1p(lift)

            for rec_item in new_items:
                matching_rules.append({
                    "recommended_item": rec_item,
                    "trigger_antecedent": sorted(list(antecedent)),
                    "trigger_antecedent_str": ", ".join(sorted(list(antecedent))),
                    "support": float(row.get("support", 0.0)),
                    "confidence": conf,
                    "lift": lift,
                    "leverage": float(row.get("leverage", 0.0)),
                    "conviction": float(row.get("conviction", 1.0)) if not pd.isna(row.get("conviction")) else None,
                    "score": score,
                    "is_fallback": False,
                })

        if not matching_rules:
            return self._fallback_recommendations(basket, top_k)

        rec_df = pd.DataFrame(matching_rules)

        # In case multiple rules recommend the same item, keep the strongest rule
        if ranking_metric == "lift":
            rec_df = rec_df.sort_values(by="lift", ascending=False)
        elif ranking_metric == "score":
            rec_df = rec_df.sort_values(by="score", ascending=False)
        else:
            rec_df = rec_df.sort_values(by="confidence", ascending=False)

        deduped = rec_df.drop_duplicates(subset=["recommended_item"]).head(top_k)

        recommendations = []
        for _, r in deduped.iterrows():
            ant_str = r["trigger_antecedent_str"]
            item_name = r["recommended_item"]
            conf_pct = r["confidence"] * 100
            lift_val = r["lift"]

            rationale = (
                f"Customers ordering '{ant_str}' also chose '{item_name}' in "
                f"{conf_pct:.1f}% of historical transactions (Lift = {lift_val:.2f}x). "
                "Suggesting this combo cross-sell creates a natural value-meal bundle."
            )

            recommendations.append({
                "item": item_name,
                "triggered_by": ant_str,
                "support": round(r["support"], 4),
                "confidence": round(r["confidence"], 4),
                "lift": round(r["lift"], 4),
                "leverage": round(r["leverage"], 4),
                "score": round(r["score"], 4),
                "rationale": rationale,
                "is_fallback": False,
            })

        return recommendations

    def _fallback_recommendations(self, basket: List[str], top_k: int) -> List[Dict[str, Any]]:
        """
        Fallback strategy when no association rules match the customer's basket.
        Recommends top overall popular items not already present.
        """
        basket_set = set(str(item).strip() for item in basket)
        recommendations = []

        if not self.top_items:
            return []

        for item, count in self.top_items.items():
            if item in basket_set:
                continue
            recommendations.append({
                "item": item,
                "triggered_by": "Overall Popularity (Fallback)",
                "support": round(count / max(sum(self.top_items.values()), 1), 4),
                "confidence": 0.0,
                "lift": 1.0,
                "leverage": 0.0,
                "score": 0.0,
                "rationale": (
                    f"'{item}' is an overall bestseller ({count} total orders). "
                    "Offered as a popular menu staple because no rule matched this exact basket."
                ),
                "is_fallback": True,
            })
            if len(recommendations) >= top_k:
                break

        return recommendations
