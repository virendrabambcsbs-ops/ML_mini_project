"""
QuickBite - Market Basket Intelligence
Module: Manual Apriori Algorithm Implementation (src/apriori_manual.py)

MANDATORY ACADEMIC REQUIREMENT:
Implemented 100% from first principles without calling mlxtend or any external
association mining libraries.

Designed for undergraduate academic viva explanations:
Includes clear algorithmic explanations, Apriori property pruning,
support counting, candidate generation, and association rule metrics.
"""

from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Any
import pandas as pd
import numpy as np


class ManualApriori:
    """
    First-principles implementation of the Apriori Algorithm & Association Rule Miner.
    
    Mathematical Foundations:
    1. Support(X) = count(X) / N
       Fraction of total transactions containing itemset X.
       
    2. Apriori Principle (Anti-monotone property of support):
       All non-empty subsets of a frequent itemset must also be frequent.
       Conversely, if an itemset is infrequent, all its supersets are pruned.
       
    3. Confidence(A -> C) = Support(A U C) / Support(A)
       Conditional probability that transaction contains C given it contains A.
       
    4. Lift(A -> C) = Confidence(A -> C) / Support(C) = Support(A U C) / (Support(A) * Support(C))
       Ratio of observed co-occurrence to expected co-occurrence under independence.
       - Lift > 1: Positive association (items complement each other).
       - Lift = 1: Independent purchase.
       - Lift < 1: Negative association (substitute items).
       
    5. Leverage(A -> C) = Support(A U C) - (Support(A) * Support(C))
       Measures the difference of X and Y appearing together in the data set
       compared to what would be expected if X and Y were statistically independent.
       
    6. Conviction(A -> C) = (1 - Support(C)) / (1 - Confidence(A -> C))
       Measures the degree of implication of the rule. Conviction = inf when Confidence = 1.
    """

    def __init__(
        self,
        min_support: float = 0.05,
        min_confidence: float = 0.20,
        min_lift: float = 1.0,
        max_len: Optional[int] = None,
    ):
        """
        Initialize the ManualApriori miner.
        
        Args:
            min_support: Minimum support threshold (between 0.0 and 1.0).
            min_confidence: Minimum confidence threshold (between 0.0 and 1.0).
            min_lift: Minimum lift threshold (>= 0.0).
            max_len: Maximum length of frequent itemsets to discover.
        """
        if not (0.0 <= min_support <= 1.0):
            raise ValueError(f"min_support must be in [0.0, 1.0], got {min_support}")
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(f"min_confidence must be in [0.0, 1.0], got {min_confidence}")
        if min_lift < 0.0:
            raise ValueError(f"min_lift must be >= 0.0, got {min_lift}")

        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_len = max_len

        # State storage
        self.itemset_support_: Dict[FrozenSet[str], float] = {}
        self.frequent_itemsets_: List[Dict[str, Any]] = []
        self.association_rules_: List[Dict[str, Any]] = []
        self.transaction_count_: int = 0
        self.itemset_counts_by_size_: Dict[int, int] = {}

    def _normalize_transactions(self, transactions: List[List[str]]) -> List[FrozenSet[str]]:
        """
        Step 1: Convert raw transaction lists into frozen sets of clean string tokens.
        Eliminates duplicate items within a single basket and enables fast subset tests.
        """
        normalized = []
        for basket in transactions:
            cleaned_basket = frozenset(
                str(item).strip() for item in basket if str(item).strip() not in ("", "nan", "None")
            )
            if cleaned_basket:
                normalized.append(cleaned_basket)
        return normalized

    def _generate_c1(self, transactions: List[FrozenSet[str]]) -> Set[FrozenSet[str]]:
        """
        Step 2: Generate C1 (candidate 1-itemsets).
        Extracts all unique individual items across all transactions.
        """
        c1: Set[FrozenSet[str]] = set()
        for basket in transactions:
            for item in basket:
                c1.add(frozenset([item]))
        return c1

    def _filter_candidates(
        self,
        candidates: Set[FrozenSet[str]],
        transactions: List[FrozenSet[str]],
        total_transactions: int,
    ) -> Dict[FrozenSet[str], float]:
        """
        Step 3 & 4: Support counting and frequent filtering.
        Counts how many transactions contain each candidate itemset.
        Retains only candidates whose support >= min_support.
        """
        counts: Dict[FrozenSet[str], int] = {cand: 0 for cand in candidates}

        for basket in transactions:
            for cand in candidates:
                if cand.issubset(basket):
                    counts[cand] += 1

        frequent_itemsets: Dict[FrozenSet[str], float] = {}
        for cand, count in counts.items():
            support = count / total_transactions
            if support >= self.min_support:
                frequent_itemsets[cand] = support

        return frequent_itemsets

    def _generate_ck(
        self,
        lk_minus_1: List[FrozenSet[str]],
        k: int,
    ) -> Set[FrozenSet[str]]:
        """
        Step 5 & 6: Candidate generation (join step) and Apriori subset-pruning.
        
        Join Step:
        Joins two frequent (k-1)-itemsets if their union has size k.
        
        Prune Step (Apriori Anti-monotone property):
        A candidate k-itemset is kept ONLY IF all of its (k-1) subsets
        are members of the previous frequent itemset set L_{k-1}.
        If any subset is infrequent, the candidate cannot be frequent and is pruned!
        """
        ck: Set[FrozenSet[str]] = set()
        lk_set = set(lk_minus_1)
        num_itemsets = len(lk_minus_1)

        # Join step: join pairs of (k-1)-itemsets
        for i in range(num_itemsets):
            for j in range(i + 1, num_itemsets):
                itemset1 = lk_minus_1[i]
                itemset2 = lk_minus_1[j]
                union_set = itemset1 | itemset2

                if len(union_set) == k:
                    # Apriori Pruning Step: Check if all (k-1)-subsets are in lk_set
                    all_subsets_frequent = True
                    for subset in combinations(union_set, k - 1):
                        if frozenset(subset) not in lk_set:
                            all_subsets_frequent = False
                            break

                    if all_subsets_frequent:
                        ck.add(union_set)

        return ck

    def fit(self, transactions: List[List[str]]) -> "ManualApriori":
        """
        Execute the full Apriori mining loop from first principles.
        """
        normalized_trans = self._normalize_transactions(transactions)
        self.transaction_count_ = len(normalized_trans)

        if self.transaction_count_ == 0:
            self.itemset_support_ = {}
            self.frequent_itemsets_ = []
            self.association_rules_ = []
            return self

        self.itemset_support_ = {}
        self.frequent_itemsets_ = []
        self.itemset_counts_by_size_ = {}

        # k = 1: Single item candidates
        c1 = self._generate_c1(normalized_trans)
        l1 = self._filter_candidates(c1, normalized_trans, self.transaction_count_)
        
        if not l1:
            return self

        # Record L1
        current_l = list(l1.keys())
        self.itemset_support_.update(l1)
        self.itemset_counts_by_size_[1] = len(l1)

        k = 2
        while current_l:
            if self.max_len is not None and k > self.max_len:
                break

            # Generate C_k with join + Apriori pruning
            ck = self._generate_ck(current_l, k)
            if not ck:
                break

            # Count support and filter L_k
            lk = self._filter_candidates(ck, normalized_trans, self.transaction_count_)
            if not lk:
                break

            self.itemset_support_.update(lk)
            self.itemset_counts_by_size_[k] = len(lk)
            current_l = list(lk.keys())
            k += 1

        # Format frequent itemsets summary
        for itemset, support in self.itemset_support_.items():
            self.frequent_itemsets_.append({
                "itemsets": itemset,
                "items": sorted(list(itemset)),
                "length": len(itemset),
                "support": round(support, 6),
                "count": int(round(support * self.transaction_count_)),
            })

        # Sort frequent itemsets by descending support
        self.frequent_itemsets_.sort(key=lambda x: (x["length"], -x["support"]))

        # Generate association rules from discovered frequent itemsets
        self._generate_rules()
        return self

    def _generate_rules(self) -> None:
        """
        Step 8 & 9: Generate association rules A -> C for all frequent itemsets of size >= 2.
        Calculates Support, Confidence, Lift, Leverage, and Conviction.
        """
        self.association_rules_ = []

        # Only itemsets with at least 2 items can form rules
        candidate_itemsets = [
            itemset for itemset in self.itemset_support_.keys() if len(itemset) >= 2
        ]

        for itemset in candidate_itemsets:
            itemset_support = self.itemset_support_[itemset]
            itemset_list = list(itemset)

            # Generate all non-empty proper subsets as antecedents
            for r in range(1, len(itemset)):
                for antecedent_tuple in combinations(itemset_list, r):
                    antecedent = frozenset(antecedent_tuple)
                    consequent = itemset - antecedent

                    if not consequent:
                        continue

                    antecedent_support = self.itemset_support_.get(antecedent)
                    consequent_support = self.itemset_support_.get(consequent)

                    # Both antecedent and consequent must be frequent itemsets
                    if antecedent_support is None or consequent_support is None:
                        continue

                    # Confidence = Support(A U C) / Support(A)
                    confidence = itemset_support / antecedent_support

                    # Apply min_confidence filter
                    if confidence < self.min_confidence:
                        continue

                    # Lift = Confidence / Support(C)
                    lift = confidence / consequent_support

                    # Apply min_lift filter
                    if lift < self.min_lift:
                        continue

                    # Leverage = Support(A U C) - (Support(A) * Support(C))
                    leverage = itemset_support - (antecedent_support * consequent_support)

                    # Conviction = (1 - Support(C)) / (1 - Confidence)
                    # Handled when Confidence -> 1.0 (conviction tends to infinity)
                    if abs(confidence - 1.0) < 1e-9:
                        conviction = float("inf")
                    else:
                        conviction = (1.0 - consequent_support) / (1.0 - confidence)

                    self.association_rules_.append({
                        "antecedents": antecedent,
                        "consequents": consequent,
                        "antecedent_str": ", ".join(sorted(list(antecedent))),
                        "consequent_str": ", ".join(sorted(list(consequent))),
                        "antecedent_len": len(antecedent),
                        "consequent_len": len(consequent),
                        "antecedent_support": round(antecedent_support, 6),
                        "consequent_support": round(consequent_support, 6),
                        "support": round(itemset_support, 6),
                        "confidence": round(confidence, 6),
                        "lift": round(lift, 6),
                        "leverage": round(leverage, 6),
                        "conviction": round(conviction, 6) if conviction != float("inf") else np.nan,
                    })

        # Sort association rules by descending lift, then confidence
        self.association_rules_.sort(key=lambda r: (-r["lift"], -r["confidence"]))

    def get_frequent_itemsets_df(self) -> pd.DataFrame:
        """
        Return frequent itemsets formatted as a clean pandas DataFrame.
        """
        if not self.frequent_itemsets_:
            return pd.DataFrame(columns=["items", "length", "support", "count"])
        
        df = pd.DataFrame(self.frequent_itemsets_)
        # Provide format compatible with mlxtend representation
        df["items_str"] = df["items"].apply(lambda x: ", ".join(x))
        return df

    def get_association_rules_df(self) -> pd.DataFrame:
        """
        Return discovered association rules as a pandas DataFrame.
        """
        columns = [
            "antecedent_str", "consequent_str", "support", "confidence",
            "lift", "leverage", "conviction", "antecedent_support", "consequent_support"
        ]
        if not self.association_rules_:
            return pd.DataFrame(columns=columns)

        df = pd.DataFrame(self.association_rules_)
        return df


def explain_manual_worked_example() -> Dict[str, Any]:
    """
    Returns a fully worked step-by-step hypothetical numerical example for academic viva.
    Total Transactions (N) = 100
    Transactions with Burger = 40
    Transactions with Fries = 30
    Transactions with both (Burger & Fries) = 20
    """
    n = 100
    count_burger = 40
    count_fries = 30
    count_both = 20

    sup_burger = count_burger / n
    sup_fries = count_fries / n
    sup_both = count_both / n
    conf_burger_to_fries = sup_both / sup_burger
    lift_burger_to_fries = conf_burger_to_fries / sup_fries
    leverage = sup_both - (sup_burger * sup_fries)
    conviction = (1 - sup_fries) / (1 - conf_burger_to_fries)

    return {
        "N": n,
        "count_A": count_burger,
        "count_B": count_fries,
        "count_AB": count_both,
        "support_A": sup_burger,
        "support_B": sup_fries,
        "support_AB": sup_both,
        "confidence_A_to_B": round(conf_burger_to_fries, 4),
        "lift_A_to_B": round(lift_burger_to_fries, 4),
        "leverage": round(leverage, 4),
        "conviction": round(conviction, 4),
        "interpretation": (
            f"Support = {sup_both:.2f} (20% of all orders contain both Burger and Fries). "
            f"Confidence = {conf_burger_to_fries:.2f} (50% of customers ordering a Burger also add Fries). "
            f"Lift = {lift_burger_to_fries:.2f} (Customers ordering a Burger are {lift_burger_to_fries:.2f}x "
            "more likely to buy Fries compared to a random customer). "
            "Because Lift > 1.0, there is a strong positive cross-selling affinity."
        )
    }
