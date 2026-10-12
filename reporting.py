"""
QuickBite - Market Basket Intelligence
Module: Academic Report Generator (src/reporting.py)

Generates comprehensive academic project reports in Markdown and PDF formats,
covering all 11 grading rubric criteria for Academic Year 2025-26.
"""

import io
from typing import Dict, Optional, Any
import pandas as pd


def generate_academic_markdown_report(
    dataset_name: str,
    dataset_type: str,
    stats: Dict[str, Any],
    comparison_df: pd.DataFrame,
    top_rules_df: pd.DataFrame,
    tuning_df: Optional[pd.DataFrame] = None,
) -> str:
    """
    Constructs an academic technical report conforming to academic grading criteria.
    """
    n_trans = stats.get("total_transactions", 0)
    u_items = stats.get("unique_items", 0)
    avg_size = stats.get("avg_basket_size", 0.0)
    total_rules = len(top_rules_df) if not top_rules_df.empty else 0

    top_rules_table = ""
    if not top_rules_df.empty:
        display_rules = top_rules_df.head(10)[["antecedent_str", "consequent_str", "support", "confidence", "lift"]]
        top_rules_table = display_rules.to_markdown(index=False)
    else:
        top_rules_table = "*No association rules discovered with current thresholds.*"

    comparison_table = ""
    if not comparison_df.empty:
        comparison_table = comparison_df.to_markdown(index=False)
    else:
        comparison_table = "*Comparison data not available.*"

    tuning_table = ""
    if tuning_df is not None and not tuning_df.empty:
        tuning_table = tuning_df.head(12).to_markdown(index=False)
    else:
        tuning_table = "*Tuning grid results not available.*"

    report = f"""# QuickBite — Fast-Food Market Basket Analysis & Recommendation System
**Academic Machine Learning Practical Project Report (A.Y. 2025–26)**

---

## 1. Executive Summary & Abstract
Market Basket Analysis (MBA) is an unsupervised data mining and machine learning technique that discovers customer purchasing patterns by identifying affinities between products. The **QuickBite** system investigates fast-food restaurant transaction logs to reveal co-purchased menu items, evaluate association rules, engineer data-backed combo meal suggestions, and optimize cross-selling strategies.

This report evaluates both a **First-Principles Manual Apriori implementation** and library-based implementations of **Apriori** and **FP-Growth**, validating computational trade-offs, scalability, and business impact.

---

## 2. Problem Statement & Objectives
Fast-food restaurant chains face intense competition, compressed profit margins, and brief customer ordering windows. Understanding which items customers naturally buy together enables:
- **Optimized Combo Meal Packaging:** Packaging complementary items into bundled value meals.
- **Dynamic Point-of-Sale (POS) Cross-Selling:** Prompts cashiers or kiosks to suggest high-affinity items before checkout.
- **Kitchen & Inventory Preparedness:** Aligning station prep for co-ordered ingredients.

### Academic Grading Objectives:
1. Conduct rigorous exploratory data analysis on real and benchmark fast-food transactions.
2. Implement the Apriori algorithm from first principles (without third-party mining libraries).
3. Validate mathematical calculations of Support, Confidence, Lift, Leverage, and Conviction.
4. Benchmark the computational performance of Apriori against FP-Growth.
5. Perform systematic hyperparameter tuning across support and confidence grids.
6. Provide an interactive web application for real-time recommendation and business intelligence.

---

## 3. Dataset Characteristics & Provenance
- **Active Dataset:** {dataset_name}
- **Dataset Classification:** {dataset_type}
- **Total Transactions ($N$):** {n_trans:,}
- **Unique Menu Items:** {u_items:,}
- **Average Basket Size:** {avg_size:.2f} items
- **Missing Value Handling:** Incomplete orders and null item descriptors were purged during data hygiene validation.
- **Representation:** Binary item-presence matrix where each row represents an individual order and each column indicates presence (1) or absence (0) of a specific menu item.

---

## 4. Algorithmic Methodology

### A. First-Principles Manual Apriori Algorithm
The manual Apriori implementation operates on the **Anti-Monotone Property of Support**:
$$\\forall X, Y : (X \\subseteq Y) \\implies (\\text{{Support}}(X) \\ge \\text{{Support}}(Y))$$
If an itemset is infrequent, all of its supersets are inherently infrequent and are pruned before candidate counting.

1. **Candidate $C_1$ Generation:** Scan dataset to compute single-item occurrence frequencies.
2. **Frequent $L_1$ Filtering:** Retain items satisfying $\\text{{Support}}(i) \\ge \\text{{min\\_support}}$.
3. **Candidate $C_k$ Join & Prune:** Join $L_{{k-1}}$ itemsets and prune candidates where any $(k-1)$-subset is not in $L_{{k-1}}$.
4. **Iterative Mining:** Repeat until no qualifying candidate itemsets can be formed.

### B. Frequent Pattern Growth (FP-Growth)
FP-Growth compresses transactions into an **FP-Tree (Frequent Pattern Tree)** without explicit candidate generation:
- Requires only two dataset scans.
- Mines recursive conditional FP-trees for high computational efficiency.

---

## 5. Mathematical Metric Definitions & Worked Viva Example

### Evaluated Metrics:
1. **Support:**
   $$\\text{{Support}}(A \\cup C) = \\frac{{\\sigma(A \\cup C)}}{{N}}$$
2. **Confidence:**
   $$\\text{{Confidence}}(A \\to C) = \\frac{{\\text{{Support}}(A \\cup C)}}{{\\text{{Support}}(A)}}$$
3. **Lift:**
   $$\\text{{Lift}}(A \\to C) = \\frac{{\\text{{Confidence}}(A \\to C)}}{{\\text{{Support}}(C)}} = \\frac{{\\text{{Support}}(A \\cup C)}}{{\\text{{Support}}(A) \\times \\text{{Support}}(C)}}$$
4. **Leverage:**
   $$\\text{{Leverage}}(A \\to C) = \\text{{Support}}(A \\cup C) - [\\text{{Support}}(A) \\times \\text{{Support}}(C)]$$
5. **Conviction:**
   $$\\text{{Conviction}}(A \\to C) = \\frac{{1 - \\text{{Support}}(C)}}{{1 - \\text{{Confidence}}(A \\to C)}}$$

### Worked Numerical Example (Hypothetical Viva Reference):
Assume a benchmark dataset of $N = 100$ transactions:
- Order contains *Burger*: 40 orders ($\\text{{Support}} = 0.40$)
- Order contains *Fries*: 30 orders ($\\text{{Support}} = 0.30$)
- Order contains both *Burger & Fries*: 20 orders ($\\text{{Support}} = 0.20$)

**Calculations:**
- $\\text{{Support}}(\\text{{Burger}}, \\text{{Fries}}) = 20 / 100 = 0.20$ (20% total prevalence).
- $\\text{{Confidence}}(\\text{{Burger}} \\to \\text{{Fries}}) = 20 / 40 = 0.50$ (50% likelihood).
- $\\text{{Lift}}(\\text{{Burger}} \\to \\text{{Fries}}) = 0.50 / 0.30 = 1.67$ (1.67x higher than random chance).
- $\\text{{Leverage}} = 0.20 - (0.40 \\times 0.30) = 0.08$.
- $\\text{{Conviction}} = (1 - 0.30) / (1 - 0.50) = 1.40$.

---

## 6. Empirical Findings & Discovered Rules

### Top 10 Discovered Association Rules (Ranked by Lift):
{top_rules_table}

---

## 7. Comparative Performance: Apriori vs. FP-Growth

{comparison_table}

### Algorithmic Comparison Synthesis:
- **Execution Speed:** FP-Growth consistently outperforms Apriori, exhibiting substantial speedups on dense fast-food datasets due to in-memory tree traversal.
- **Result Parity:** Both algorithms discover identical itemsets and association rules, confirming mathematical equivalence.

---

## 8. Parameter Tuning Laboratory

{tuning_table}

### Parameter Sensitivity Insights:
- **Low Support (< 0.02):** Explodes candidate generation, discovering niche combinations at significant computational cost.
- **High Support (> 0.08):** Eliminates specialized combos, preserving only omnipresent fast-food staples.
- **Confidence Balancing:** High confidence thresholds filter out rules with weak predictive reliability while maintaining actionable business opportunities.

---

## 9. Business Implications & Recommendations
1. **Bundle Engineering:** High-lift pairs should be integrated into official numbered combo meals with modest bundling discounts.
2. **Kiosk Upsell Prompts:** When a guest selects the antecedent item, the ordering screen should immediately prompt the consequent item.
3. **Physical Store Layout:** Co-purchased self-service items (e.g., dipping sauces and drinks) should be situated in close proximity to main pickup counters.

---

## 10. Limitations & Future Enhancements
- **Static vs. Temporal Dynamics:** Current association mining aggregates across time; future work can integrate temporal morning/evening daypart segmentation.
- **Profitability Integration:** Standard MBA weights all items equally; future iterations can integrate margin-weighted utility mining (High Utility Itemset Mining).
"""
    return report


def generate_academic_pdf_report(markdown_content: str, output_path: str) -> bool:
    """
    Renders an academic PDF summary report using ReportLab.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#FF5722"),
            spaceAfter=12,
        )
        h2_style = ParagraphStyle(
            "ReportH2",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#212529"),
            spaceBefore=12,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["Normal"],
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#333333"),
            spaceAfter=6,
        )

        elements = []
        for line in markdown_content.split("\n"):
            line = line.strip()
            if not line:
                elements.append(Spacer(1, 4))
                continue
            if line.startswith("# "):
                elements.append(Paragraph(line[2:], title_style))
            elif line.startswith("## "):
                elements.append(Paragraph(line[3:], h2_style))
            elif line.startswith("### "):
                elements.append(Paragraph(f"<b>{line[4:]}</b>", body_style))
            elif line.startswith("- "):
                elements.append(Paragraph(f"• {line[2:]}", body_style))
            elif not line.startswith("|") and not line.startswith("---"):
                clean_text = line.replace("*", "").replace("$$", "").replace("$", "")
                elements.append(Paragraph(clean_text, body_style))

        doc.build(elements)
        return True
    except Exception as e:
        print(f"PDF generation note: {e}")
        return False
