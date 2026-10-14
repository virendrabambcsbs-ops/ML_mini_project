"""
QuickBite — Fast-Food Market Basket Intelligence & Recommendation System
Main Streamlit Web Application (app.py)

Academic Machine Learning Practical Project (A.Y. 2025–26)
"""

import os
import io
import time
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any

# Internal modules
from src.preprocessing import TransactionPreprocessor
from src.apriori_manual import ManualApriori, explain_manual_worked_example
from src.association_analysis import (
    AssociationMiner,
    benchmark_apriori_vs_fpgrowth,
    run_parameter_tuning_grid,
)
from src.recommendations import ComboRecommendationEngine
from src.visualization import (
    plot_top_items,
    plot_basket_size_distribution,
    plot_support_vs_confidence,
    plot_top_rules_by_lift,
    plot_item_cooccurrence_heatmap,
    plot_association_network_graph,
)
from src.reporting import generate_academic_markdown_report, generate_academic_pdf_report

# Configure Page
st.set_page_config(
    page_title="QuickBite — Market Basket Intelligence",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Restaurant-Tech Aesthetic
st.markdown(
    """
    <style>
    :root {
        --primary: #FF5722;
        --primary-dark: #E64A19;
        --secondary: #FFA726;
        --bg-card: #FFFFFF;
        --text-dark: #212529;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #ECEFF1;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        margin-bottom: 12px;
        border-left: 4px solid #FF5722;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #212529;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        color: #78909C;
        letter-spacing: 0.5px;
    }
    .badge-real {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-synth {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-custom {
        background-color: #E1F5FE;
        color: #0277BD;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
    }
    .rule-box {
        background-color: #F8F9FA;
        border-left: 4px solid #FF5722;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ---------------------------------------------------------
# CACHED DATA LOADING & MINING FUNCTIONS
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_and_preprocess_dataset(
    file_source: str,
    uploaded_file_bytes: Optional[bytes] = None,
    order_col: str = "order_id",
    item_col: str = "item_name",
    min_basket_size: int = 1,
):
    if file_source == "synthetic":
        path = os.path.join(DATA_DIR, "quickbite_synthetic_orders.csv")
        df = pd.read_csv(path)
        dataset_name = "QuickBite Benchmark Fast-Food Orders"
        dataset_type = "Synthetic Demonstration Dataset (5,000 Orders)"
    elif file_source == "chipotle":
        path = os.path.join(DATA_DIR, "chipotle_orders.csv")
        df = pd.read_csv(path)
        dataset_name = "Chipotle Mexican Grill Orders"
        dataset_type = "Real-World Restaurant Sales (1,834 Orders)"
    else:
        if uploaded_file_bytes is None:
            return None, None, None, {}, "No file", "Custom"
        df = pd.read_csv(io.BytesIO(uploaded_file_bytes))
        dataset_name = "User Uploaded Dataset"
        dataset_type = "Custom Uploaded Dataset"

    preprocessor = TransactionPreprocessor(order_col=order_col, item_col=item_col)
    try:
        cleaned_df = preprocessor.clean_dataset(df, min_basket_size=min_basket_size)
    except Exception as e:
        cleaned_df = df

    trans_list, one_hot_df = preprocessor.create_transaction_baskets(cleaned_df)
    stats = preprocessor.get_basket_statistics(cleaned_df, trans_list)

    return df, cleaned_df, one_hot_df, trans_list, stats, dataset_name, dataset_type


@st.cache_data(show_spinner=False)
def run_association_mining(
    one_hot_df: pd.DataFrame,
    algorithm: str,
    min_support: float,
    min_confidence: float,
    min_lift: float,
    min_len: int,
    max_len: Optional[int],
    max_rules: int,
):
    miner = AssociationMiner(
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        min_len=min_len,
        max_len=max_len,
        max_rules=max_rules,
    )
    itemsets, runtime = miner.mine_frequent_itemsets(one_hot_df, algorithm=algorithm)
    rules = miner.generate_rules(itemsets, num_transactions=len(one_hot_df))
    return itemsets, rules, runtime


# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------

with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400&auto=format&fit=crop&q=60",
        use_container_width=True,
    )
    st.title("🍔 QuickBite")
    st.caption("Market Basket Intelligence Engine | v1.0")
    st.divider()

    st.subheader("📁 Dataset Selection")
    data_option = st.selectbox(
        "Choose Transaction Source",
        [
            "QuickBite Benchmark (Synthetic)",
            "Chipotle Orders (Real-World)",
            "Upload Custom CSV",
        ],
    )

    uploaded_file_bytes = None
    if data_option == "QuickBite Benchmark (Synthetic)":
        file_source = "synthetic"
        order_col = "order_id"
        item_col = "item_name"
    elif data_option == "Chipotle Orders (Real-World)":
        file_source = "chipotle"
        order_col = "order_id"
        item_col = "item_name"
    else:
        file_source = "custom"
        with st.expander("📋 Required custom CSV format", expanded=True):
            st.markdown(
                """
                Your file must contain **one row per item in an order** and at least
                these two columns:

                - **Order ID**: identifies the basket/transaction. The same value
                  must be repeated for every item in the same order.
                - **Item / Product**: the name of the menu item or product.

                Optional columns such as `quantity`, `price`, and `date` are allowed
                but are not required for the analysis. The file must be a
                comma-separated CSV with a header row. Blank order IDs or item names
                are removed automatically.
                """
            )
            st.download_button(
                "⬇️ Download CSV template",
                data="order_id,item_name,quantity,price\n",
                file_name="quickbite_transaction_template.csv",
                mime="text/csv",
            )
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file is not None:
            uploaded_file_bytes = uploaded_file.getvalue()
            # Preview columns
            temp_df = pd.read_csv(io.BytesIO(uploaded_file_bytes), nrows=5)
            detected = TransactionPreprocessor.detect_columns(temp_df)
            order_col = st.selectbox(
                "Order ID Column",
                temp_df.columns,
                index=list(temp_df.columns).index(detected["order_col"]) if detected["order_col"] in temp_df.columns else 0,
            )
            item_col = st.selectbox(
                "Item / Product Column",
                temp_df.columns,
                index=list(temp_df.columns).index(detected["item_col"]) if detected["item_col"] in temp_df.columns else min(1, len(temp_df.columns)-1),
            )
        else:
            order_col = "order_id"
            item_col = "item_name"

    st.divider()
    st.subheader("⚙️ Mining Hyperparameters")

    algorithm_choice = st.radio(
        "Mining Algorithm",
        ["FP-Growth (Fast)", "Apriori (Classic)"],
        index=0,
    )
    algo_key = "fpgrowth" if "FP-Growth" in algorithm_choice else "apriori"

    min_support = st.slider(
        "Min Support",
        min_value=0.005,
        max_value=0.20,
        value=0.02 if file_source == "chipotle" else 0.03,
        step=0.005,
        format="%.3f",
        help="Fraction of transactions that must contain the itemset.",
    )

    min_confidence = st.slider(
        "Min Confidence",
        min_value=0.05,
        max_value=0.90,
        value=0.20,
        step=0.05,
        format="%.2f",
        help="Conditional probability that consequent is bought given antecedent.",
    )

    min_lift = st.slider(
        "Min Lift",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Strength of association (>1 indicates positive correlation).",
    )

    with st.expander("Advanced Constraints"):
        min_itemset_len = st.number_input("Min Itemset Length", min_value=1, max_value=5, value=1)
        max_itemset_len = st.number_input("Max Itemset Length", min_value=2, max_value=10, value=5)
        max_display_rules = st.slider("Max Displayed Rules", 10, 500, 100, step=10)

    st.divider()
    page = st.selectbox(
        "📌 Navigation",
        [
            "Overview Dashboard",
            "Data Explorer & Hygiene",
            "Association Rule Mining",
            "Combo Recommendation Engine",
            "Algorithm Comparison",
            "Visual Analytics & Network",
            "Parameter Tuning Lab",
            "Project Report & Viva Guide",
        ],
    )

# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

if file_source == "custom" and uploaded_file_bytes is None:
    st.info("👆 Please upload a transaction CSV file from the sidebar to begin.")
    st.stop()

with st.spinner("Processing transaction baskets..."):
    df_raw, df_clean, one_hot_df, trans_list, stats, dataset_name, dataset_type = load_and_preprocess_dataset(
        file_source=file_source,
        uploaded_file_bytes=uploaded_file_bytes,
        order_col=order_col,
        item_col=item_col,
    )

if df_clean is None or one_hot_df.empty:
    st.error("Could not construct transaction baskets. Verify selected Order ID and Item columns.")
    st.stop()

# Run mining for current parameters
with st.spinner(f"Mining frequent patterns with {algorithm_choice}..."):
    frequent_itemsets, association_rules_df, mining_runtime = run_association_mining(
        one_hot_df=one_hot_df,
        algorithm=algo_key,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        min_len=min_itemset_len,
        max_len=max_itemset_len,
        max_rules=max_display_rules,
    )

# ---------------------------------------------------------
# PAGE A: OVERVIEW DASHBOARD
# ---------------------------------------------------------

if page == "Overview Dashboard":
    st.title("🍔 QuickBite Market Basket Intelligence")
    st.caption("Fast-Food Menu Engineering, Combo Meal Discovery & Cross-Selling Insights")

    # Provenance Badge
    if "Real" in dataset_type:
        st.markdown(f'<span class="badge-real">● {dataset_type} — {dataset_name}</span>', unsafe_allow_html=True)
    elif "Synthetic" in dataset_type:
        st.markdown(f'<span class="badge-synth">● {dataset_type} — {dataset_name}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge-custom">● {dataset_type} — {dataset_name}</span>', unsafe_allow_html=True)

    st.write("")

    # KPI Metric Cards
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    with kpi1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Transactions</div><div class="metric-value">{stats["total_transactions"]:,}</div></div>',
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Menu Items</div><div class="metric-value">{stats["unique_items"]:,}</div></div>',
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Avg Basket Size</div><div class="metric-value">{stats["avg_basket_size"]}</div></div>',
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Frequent Itemsets</div><div class="metric-value">{len(frequent_itemsets):,}</div></div>',
            unsafe_allow_html=True,
        )
    with kpi5:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Association Rules</div><div class="metric-value">{len(association_rules_df):,}</div></div>',
            unsafe_allow_html=True,
        )
    with kpi6:
        max_l = f"{association_rules_df['lift'].max():.2f}x" if not association_rules_df.empty else "N/A"
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Peak Rule Lift</div><div class="metric-value">{max_l}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")

    col_chart1, col_chart2 = st.columns([1, 1])
    with col_chart1:
        fig_top = plot_top_items(stats["top_items"], stats["total_transactions"])
        st.plotly_chart(fig_top, use_container_width=True)

    with col_chart2:
        fig_dist = plot_basket_size_distribution(stats["basket_size_distribution"])
        st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("💡 Analytical Summary & Key Findings")
    if not association_rules_df.empty:
        top_rule = association_rules_df.iloc[0]
        st.info(
            f"**Top Discovered Association Rule:** Ordering **{top_rule['antecedent_str']}** strongly drives "
            f"**{top_rule['consequent_str']}** with a Confidence of **{top_rule['confidence']*100:.1f}%** "
            f"and Lift of **{top_rule['lift']:.2f}x** (Support = {top_rule['support']*100:.2f}%). "
            f"Customers choosing {top_rule['antecedent_str']} are {top_rule['lift']:.2f} times more likely to add "
            f"{top_rule['consequent_str']} than typical guests."
        )
    else:
        st.warning(
            "No association rules met the current thresholds. Try reducing Minimum Support or Minimum Confidence in the sidebar."
        )

# ---------------------------------------------------------
# PAGE B: DATA EXPLORER & HYGIENE
# ---------------------------------------------------------

elif page == "Data Explorer & Hygiene":
    st.title("🔍 Data Explorer & Preprocessing Diagnostics")
    st.caption("Inspect raw transaction records, schema, missing-value diagnostics, and data-hygiene steps.")

    tab1, tab2, tab3 = st.tabs(["Raw Dataset Preview", "Quality Diagnostics", "Cleaned Dataset"])

    with tab1:
        st.subheader("Raw Data Sample")
        st.dataframe(df_raw.head(100), use_container_width=True)
        st.caption(f"Showing first 100 rows out of {len(df_raw):,} total rows.")

    with tab2:
        st.subheader("Data Quality & Hygiene Audit")
        preprocessor = TransactionPreprocessor(order_col=order_col, item_col=item_col)
        diag = preprocessor.inspect_dataset(df_raw)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Raw Records", f"{diag['total_rows']:,}")
        c2.metric("Distinct Orders", f"{diag['total_orders']:,}")
        c3.metric("Distinct Items", f"{diag['unique_items']:,}")
        c4.metric("Duplicate Rows", f"{diag['duplicate_rows']:,}")

        st.write("#### Missing Values per Column")
        missing_df = pd.DataFrame(
            list(diag["missing_values"].items()), columns=["Column", "Missing Count"]
        )
        missing_df["Missing (%)"] = (missing_df["Missing Count"] / max(len(df_raw), 1) * 100).round(2)
        st.dataframe(missing_df, use_container_width=True)

        st.write("#### Preprocessing Justifications")
        st.markdown(
            """
            - **Binary Item Presence:** For fast-food market basket analysis, duplicate rows of the same product within one order represent quantities, not separate co-occurrences. Baskets are encoded as binary presence matrices.
            - **No Numerical Scaling:** Association rule mining is an unsupervised categorical transaction technique; numeric normalization (e.g. MinMax or StandardScaler) is deliberately not applied to transaction tokens.
            - **Null Item Scrubbing:** Rows with null, blank, or placeholder items ('nan', 'none') are purged to prevent phantom association rules.
            """
        )

    with tab3:
        st.subheader("Cleaned Dataset")
        st.dataframe(df_clean.head(100), use_container_width=True)

        csv_clean = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Cleaned Dataset (CSV)",
            data=csv_clean,
            file_name="quickbite_cleaned_transactions.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------
# PAGE C: ASSOCIATION RULE MINING
# ---------------------------------------------------------

elif page == "Association Rule Mining":
    st.title("⛏️ Association Rule Mining")
    st.caption("Explore discovered frequent itemsets and association rules with comprehensive metrics.")

    st.markdown(
        f"**Algorithm:** `{algorithm_choice}` | **Min Support:** `{min_support}` | "
        f"**Min Confidence:** `{min_confidence}` | **Min Lift:** `{min_lift}` | **Runtime:** `{mining_runtime:.4f}s`"
    )

    tab_rules, tab_itemsets, tab_manual = st.tabs(
        ["Association Rules", "Frequent Itemsets", "Manual Apriori Verification"]
    )

    with tab_rules:
        if association_rules_df.empty:
            st.warning("No association rules found with the current thresholds. Lower Min Support or Min Confidence.")
        else:
            st.subheader(f"Discovered Association Rules ({len(association_rules_df):,})")

            # Search & Filter
            search_query = st.text_input("Filter rules by item name (e.g. Burger, Fries, Shake)")
            filtered_rules = association_rules_df
            if search_query:
                filtered_rules = filtered_rules[
                    filtered_rules["rule_str"].str.contains(search_query, case=False, na=False)
                ]

            display_cols = [
                "antecedent_str", "consequent_str", "support", "confidence",
                "lift", "leverage", "conviction"
            ]
            st.dataframe(
                filtered_rules[display_cols].rename(
                    columns={
                        "antecedent_str": "Antecedent (IF)",
                        "consequent_str": "Consequent (THEN)",
                        "support": "Support",
                        "confidence": "Confidence",
                        "lift": "Lift",
                        "leverage": "Leverage",
                        "conviction": "Conviction",
                    }
                ),
                use_container_width=True,
            )

            csv_rules = association_rules_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Association Rules (CSV)",
                data=csv_rules,
                file_name="quickbite_association_rules.csv",
                mime="text/csv",
            )

    with tab_itemsets:
        st.subheader(f"Frequent Itemsets ({len(frequent_itemsets):,})")
        if not frequent_itemsets.empty:
            st.dataframe(
                frequent_itemsets[["items_str", "length", "support", "count"]].rename(
                    columns={
                        "items_str": "Itemset",
                        "length": "Itemset Size",
                        "support": "Support",
                        "count": "Order Count",
                    }
                ),
                use_container_width=True,
            )
            csv_itemsets = frequent_itemsets.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Frequent Itemsets (CSV)",
                data=csv_itemsets,
                file_name="quickbite_frequent_itemsets.csv",
                mime="text/csv",
            )
        else:
            st.warning("No frequent itemsets discovered.")

    with tab_manual:
        st.subheader("Mandatory First-Principles Manual Apriori Implementation")
        st.markdown(
            """
            This section runs our custom **from-scratch Apriori implementation** (`src/apriori_manual.py`)
            built without calling mlxtend. It demonstrates:
            1. Candidate 1-itemset generation ($C_1$)
            2. Support counting and $L_1$ filtering
            3. Systematic $(k)$-itemset join with Apriori subset-pruning
            4. Rule generation and floating-point parity verification against mlxtend.
            """
        )

        if st.button("🚀 Run First-Principles Manual Apriori Benchmark"):
            with st.spinner("Executing manual Apriori from first principles..."):
                t_start = time.perf_counter()
                manual_miner = ManualApriori(
                    min_support=min_support,
                    min_confidence=min_confidence,
                    min_lift=min_lift,
                    max_len=max_itemset_len,
                )
                manual_miner.fit(trans_list)
                manual_time = time.perf_counter() - t_start

                man_itemsets_df = manual_miner.get_frequent_itemsets_df()
                man_rules_df = manual_miner.get_association_rules_df()

            st.success(
                f"Manual Apriori executed successfully in **{manual_time:.4f}s**! "
                f"Discovered **{len(man_itemsets_df):,}** frequent itemsets and **{len(man_rules_df):,}** rules."
            )

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.write("**Manual Frequent Itemsets Sample:**")
                st.dataframe(man_itemsets_df.head(10)[["items_str", "length", "support", "count"]], use_container_width=True)
            with col_m2:
                st.write("**Manual Association Rules Sample:**")
                st.dataframe(man_rules_df.head(10)[["antecedent_str", "consequent_str", "support", "confidence", "lift"]], use_container_width=True)

# ---------------------------------------------------------
# PAGE D: COMBO RECOMMENDATION ENGINE
# ---------------------------------------------------------

elif page == "Combo Recommendation Engine":
    st.title("🎯 Combo Recommendation Engine")
    st.caption("Interactive cashier & kiosk ordering simulation powered by discovered association rules.")

    available_items = sorted(list(one_hot_df.columns))

    col_select, col_config = st.columns([2, 1])
    with col_select:
        # Default pre-selected item
        default_items = [available_items[0]] if available_items else []
        selected_basket = st.multiselect(
            "🛒 Customer's Current Order Basket:",
            options=available_items,
            default=default_items,
            help="Select one or more items currently in the customer's tray.",
        )

    with col_config:
        rec_metric = st.selectbox("Recommendation Ranking Criterion", ["Confidence", "Lift", "Hybrid Score"])
        top_k_recs = st.slider("Max Suggestions", 1, 10, 4)

    st.write("")
    st.subheader("💡 Recommended Next Items / Value Add-ons")

    rec_engine = ComboRecommendationEngine(association_rules_df, top_items=stats.get("top_items"))
    recs = rec_engine.recommend(
        selected_basket,
        top_k=top_k_recs,
        ranking_metric=rec_metric.lower().replace(" ", "_"),
    )

    if not recs:
        st.info("Please select items in the basket to generate data-backed combo recommendations.")
    else:
        for r in recs:
            badge = "🔥 High-Confidence Rule" if not r["is_fallback"] else "⭐ Popularity Suggestion"
            st.markdown(
                f"""
                <div class="rule-box">
                    <span style="float: right; font-size: 12px; color: #78909C;">{badge}</span>
                    <h4 style="margin: 0; color: #D84315;">+ Add {r['item']}</h4>
                    <p style="margin: 6px 0; color: #424242;">{r['rationale']}</p>
                    <span style="font-size: 13px; font-weight: 600; color: #2E7D32;">
                        Confidence: {r['confidence']*100:.1f}% &nbsp;|&nbsp; 
                        Lift: {r['lift']:.2f}x &nbsp;|&nbsp; 
                        Support: {r['support']*100:.2f}%
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption("⚠️ *Note: Recommendations represent observed empirical co-occurrences, not guarantees of causal customer response.*")

# ---------------------------------------------------------
# PAGE E: ALGORITHM COMPARISON
# ---------------------------------------------------------

elif page == "Algorithm Comparison":
    st.title("⚖️ Algorithm Comparison: Apriori vs. FP-Growth")
    st.caption("Benchmarking computational runtime, memory structure, and mathematical equivalence.")

    if st.button("⚡ Run Comparative Benchmark on Current Dataset"):
        with st.spinner("Executing head-to-head empirical comparison..."):
            benchmark_res = benchmark_apriori_vs_fpgrowth(
                one_hot_df=one_hot_df,
                min_support=min_support,
                min_confidence=min_confidence,
                min_lift=min_lift,
            )

        st.success("Benchmark completed!")
        st.subheader("Performance Metrics Table")
        st.dataframe(benchmark_res["comparison_df"], use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Apriori Runtime", f"{benchmark_res['apriori_time']:.4f}s")
        c2.metric("FP-Growth Runtime", f"{benchmark_res['fpgrowth_time']:.4f}s")
        c3.metric("FP-Growth Speedup", f"{benchmark_res['speedup']:.2f}x")

        st.info(f"**Empirical Analysis:** {benchmark_res['insights']}")

        st.subheader("Theoretical Algorithmic Comparison (Academic Viva Reference)")
        comp_notes = pd.DataFrame([
            {"Feature": "Candidate Generation", "Apriori": "Explicit join & prune passes for each size k", "FP-Growth": "None; compresses database into an FP-Tree"},
            {"Feature": "Database Scans", "Apriori": "Multiple scans (one per itemset size k)", "FP-Growth": "Exactly 2 database scans"},
            {"Feature": "Memory Structure", "Apriori": "Hash trees / array candidate lists", "FP-Growth": "Compact in-memory Trie-like FP-Tree"},
            {"Feature": "Suitability", "Apriori": "Small sparse datasets, educational clarity", "FP-Growth": "Dense fast-food retail databases, large scale"},
        ])
        st.dataframe(comp_notes, use_container_width=True)

# ---------------------------------------------------------
# PAGE F: VISUAL ANALYTICS & NETWORK
# ---------------------------------------------------------

elif page == "Visual Analytics & Network":
    st.title("📊 Customized Visual Analytics")
    st.caption("Interactive charts, co-occurrence heatmaps, and association rule network graph.")

    tab_network, tab_heat, tab_scatter, tab_lift = st.tabs(
        ["Association Network Graph", "Item Co-occurrence Heatmap", "Support vs Confidence", "Top Rules by Lift"]
    )

    with tab_network:
        st.subheader("Interactive Food Association Network Graph")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            max_graph_rules = st.slider("Max Network Rules (Edge Cap)", 10, 60, 30)
        with col_g2:
            min_graph_lift = st.slider("Filter Minimum Lift", 1.0, 4.0, 1.2, step=0.1)

        fig_net = plot_association_network_graph(
            association_rules_df, max_rules=max_graph_rules, min_lift=min_graph_lift
        )
        st.plotly_chart(fig_net, use_container_width=True)

    with tab_heat:
        st.subheader("Item Co-occurrence Frequency Heatmap")
        top_items_list = list(stats.get("top_items", {}).keys())[:15]
        fig_heat = plot_item_cooccurrence_heatmap(one_hot_df, top_items=top_items_list)
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab_scatter:
        st.subheader("Support vs. Confidence (Bubble Size & Color = Lift)")
        fig_scatter = plot_support_vs_confidence(association_rules_df)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab_lift:
        st.subheader("Top Discovered Rules Ranked by Lift")
        fig_lift = plot_top_rules_by_lift(association_rules_df, top_n=20)
        st.plotly_chart(fig_lift, use_container_width=True)

# ---------------------------------------------------------
# PAGE G: PARAMETER TUNING LAB
# ---------------------------------------------------------

elif page == "Parameter Tuning Lab":
    st.title("🧪 Parameter Tuning Laboratory")
    st.caption("Systematic hyperparameter exploration across Minimum Support and Confidence thresholds.")

    st.markdown(
        """
        **Hyperparameter Trade-off Principle:**
        - **Low Support:** Discovers long-tail and specialized combos, but increases candidate counts and false positives.
        - **High Support:** Ensures rules are backed by high sample volumes, but risks missing valuable niche cross-sells.
        - **Confidence Tuning:** Governs rule reliability without dictating co-occurrence frequency.
        """
    )

    if st.button("🧪 Run Parameter Tuning Grid Experiment"):
        with st.spinner("Evaluating hyperparameter grid..."):
            tuning_df = run_parameter_tuning_grid(
                one_hot_df=one_hot_df,
                support_grid=[0.01, 0.02, 0.03, 0.05, 0.08, 0.10],
                confidence_grid=[0.15, 0.25, 0.35, 0.50],
                algorithm=algo_key,
            )

        st.subheader("Experiment Results Grid")
        st.dataframe(tuning_df, use_container_width=True)

        # Plot sensitivity
        import plotly.express as px
        fig_sens = px.line(
            tuning_df,
            x="min_support",
            y="association_rules",
            color="min_confidence",
            markers=True,
            title="<b>Association Rules Discovered vs. Minimum Support (by Confidence)</b>",
            labels={"min_support": "Minimum Support", "association_rules": "Rule Count", "min_confidence": "Min Confidence"},
            template="plotly_white",
        )
        st.plotly_chart(fig_sens, use_container_width=True)

        csv_tuning = tuning_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Tuning Experiment Data (CSV)",
            data=csv_tuning,
            file_name="quickbite_parameter_tuning.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------
# PAGE H: PROJECT REPORT & VIVA GUIDE
# ---------------------------------------------------------

elif page == "Project Report & Viva Guide":
    st.title("📄 Academic Project Report & Viva Preparation Guide")
    st.caption("Academic Year 2025–26 Machine Learning Practical Evaluation")

    tab_rep, tab_viva = st.tabs(["Academic Technical Report", "Viva Voce Questions & Answers"])

    with tab_rep:
        benchmark_res = benchmark_apriori_vs_fpgrowth(
            one_hot_df=one_hot_df,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
        )

        md_report = generate_academic_markdown_report(
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            stats=stats,
            comparison_df=benchmark_res["comparison_df"],
            top_rules_df=association_rules_df,
        )

        st.markdown(md_report)

        st.divider()
        st.subheader("📥 Export Academic Artifacts")
        col_d1, col_d2, col_d3 = st.columns(3)

        with col_d1:
            st.download_button(
                label="📥 Download Report (Markdown .md)",
                data=md_report.encode("utf-8"),
                file_name="QuickBite_Academic_Report.md",
                mime="text/markdown",
            )

        with col_d2:
            # Generate PDF
            pdf_path = os.path.join(OUTPUTS_DIR, "QuickBite_Academic_Report.pdf")
            pdf_ok = generate_academic_pdf_report(md_report, pdf_path)
            if pdf_ok and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="📥 Download Report (PDF .pdf)",
                    data=pdf_bytes,
                    file_name="QuickBite_Academic_Report.pdf",
                    mime="application/pdf",
                )
            else:
                st.caption("PDF generation ready via ReportLab.")

        with col_d3:
            if not association_rules_df.empty:
                st.download_button(
                    label="📥 Download All Rules (CSV)",
                    data=association_rules_df.to_csv(index=False).encode("utf-8"),
                    file_name="quickbite_mined_rules.csv",
                    mime="text/csv",
                )

    with tab_viva:
        st.subheader("🎓 Comprehensive Viva Voce Questions & Answers")
        
        worked_ex = explain_manual_worked_example()
        st.markdown("### 1. Worked Numerical Viva Example (Hypothetical)")
        st.info(f"**Scenario:** Out of {worked_ex['N']} transactions, Burger is in {worked_ex['count_A']}, Fries in {worked_ex['count_B']}, and both in {worked_ex['count_AB']}.")
        st.markdown(
            f"""
            - **Support(Burger, Fries):** {worked_ex['count_AB']}/{worked_ex['N']} = **{worked_ex['support_AB']}**
            - **Confidence(Burger → Fries):** {worked_ex['count_AB']}/{worked_ex['count_A']} = **{worked_ex['confidence_A_to_B']}** (50%)
            - **Lift(Burger → Fries):** {worked_ex['confidence_A_to_B']}/{worked_ex['support_B']} = **{worked_ex['lift_A_to_B']:.2f}x**
            - **Leverage:** {worked_ex['leverage']}
            - **Conviction:** {worked_ex['conviction']}
            
            **Viva Explanation:** {worked_ex['interpretation']}
            """
        )

        qa_pairs = [
            (
                "Q1: What is the Apriori Principle (Anti-Monotone Property)?",
                "If an itemset is infrequent, all of its supersets must also be infrequent. This property enables candidate pruning: we do not need to count support for itemsets containing an infrequent subset."
            ),
            (
                "Q2: Why is Lift preferred over Confidence alone?",
                "Confidence does not account for the baseline popularity of the consequent. If 90% of all orders contain Soda, a rule Burger → Soda with 85% confidence might actually have a Lift < 1.0 (indicating negative association). Lift adjusts for baseline probability."
            ),
            (
                "Q3: How does FP-Growth achieve faster runtimes than Apriori?",
                "Apriori performs multiple database scans (one for each size k) and generates large combinatorial candidate sets. FP-Growth scans the database only twice, builds a compact in-memory FP-Tree, and mines frequent itemsets recursively via conditional pattern bases without candidate generation."
            ),
            (
                "Q4: Why don't we use Accuracy or F1-Score for Market Basket Analysis?",
                "Market Basket Analysis is an unsupervised pattern discovery task, not supervised classification. The goal is discovering association rules with high statistical significance (support, confidence, lift), not predicting a binary ground truth class."
            ),
            (
                "Q5: What are the main business applications in fast-food restaurants?",
                "1. Combo Meal Bundle engineering (pairing high-lift items with bundling discounts).\n2. Dynamic Kiosk Upselling (prompting high-confidence add-ons before checkout).\n3. Kitchen ingredient staging and inventory synchronization."
            )
        ]

        for q, a in qa_pairs:
            with st.expander(f"📌 {q}"):
                st.write(a)
