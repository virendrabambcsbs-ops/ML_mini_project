"""
QuickBite - Market Basket Intelligence
Module: Customized Visual Analytics (src/visualization.py)

Generates responsive, publication-quality Plotly figures for market basket analytics:
- Top Items Bar Chart
- Basket Size Distribution
- Support vs Confidence Scatter Plot
- Lift Distribution
- Item Co-occurrence Heatmap
- Association Network Graph (NetworkX + Plotly)
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# QuickBite Aesthetic Color Palette
PRIMARY_COLOR = "#FF5722"  # Deep Vibrant Orange
SECONDARY_COLOR = "#FF9800" # Warm Amber
DARK_BG = "#1E1E24"
LIGHT_BG = "#F8F9FA"
ACCENT_BLUE = "#0288D1"
ACCENT_GREEN = "#2E7D32"


def plot_top_items(top_items_dict: Dict[str, int], total_transactions: int) -> go.Figure:
    """
    Horizontal bar chart of top fast-food menu items with transaction count and support %.
    """
    if not top_items_dict:
        fig = go.Figure()
        fig.update_layout(title="No item data available")
        return fig

    df = pd.DataFrame(list(top_items_dict.items()), columns=["Item", "Count"])
    df["Support (%)"] = (df["Count"] / max(total_transactions, 1) * 100).round(2)
    df = df.sort_values(by="Count", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df["Count"],
            y=df["Item"],
            orientation="h",
            marker=dict(
                color=df["Count"],
                colorscale="Oranges",
                showscale=False,
                line=dict(color="#D84315", width=1),
            ),
            text=[f"{cnt:,} ({sup}%)" for cnt, sup in zip(df["Count"], df["Support (%)"])],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Orders: %{x:,}<br>Support: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Top 15 Most Frequently Ordered Menu Items</b>",
        xaxis_title="Order Frequency (Transactions)",
        yaxis_title="Food Item",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=450,
    )
    return fig


def plot_basket_size_distribution(size_distribution_dict: Dict[int, int]) -> go.Figure:
    """
    Bar chart showing the distribution of basket sizes (items per order).
    """
    if not size_distribution_dict:
        fig = go.Figure()
        fig.update_layout(title="No transaction data available")
        return fig

    df = pd.DataFrame(list(size_distribution_dict.items()), columns=["Basket Size", "Order Count"])
    df = df.sort_values(by="Basket Size")

    fig = go.Figure(
        go.Bar(
            x=df["Basket Size"].astype(str),
            y=df["Order Count"],
            marker=dict(color="#FF7043", line=dict(color="#BF360C", width=1)),
            text=[f"{v:,}" for v in df["Order Count"]],
            textposition="outside",
            hovertemplate="Basket Size: %{x} items<br>Orders: %{y:,}<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Transaction Basket Size Distribution</b>",
        xaxis_title="Number of Items in Basket",
        yaxis_title="Number of Transactions",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=380,
    )
    return fig


def plot_support_vs_confidence(rules_df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot of Support vs Confidence with Lift as color scale and bubble size.
    """
    if rules_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No association rules available")
        return fig

    df = rules_df.copy()
    max_bubble = 25
    min_bubble = 8

    # Size proportional to lift
    lift_vals = df["lift"].fillna(1.0)
    sizes = np.interp(lift_vals, (lift_vals.min(), lift_vals.max()), (min_bubble, max_bubble))

    fig = go.Figure(
        go.Scatter(
            x=df["support"],
            y=df["confidence"],
            mode="markers",
            marker=dict(
                size=sizes,
                color=df["lift"],
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(title="Lift"),
                opacity=0.85,
                line=dict(color="white", width=1),
            ),
            text=df["rule_str"],
            hovertemplate=(
                "<b>Rule:</b> %{text}<br>"
                "<b>Support:</b> %{x:.4f}<br>"
                "<b>Confidence:</b> %{y:.4f}<br>"
                "<b>Lift:</b> %{marker.color:.2f}x<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="<b>Association Rules: Support vs. Confidence (Colored by Lift)</b>",
        xaxis_title="Support (Prevalence across all orders)",
        yaxis_title="Confidence (Probability of Consequent given Antecedent)",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=450,
    )
    return fig


def plot_top_rules_by_lift(rules_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Horizontal bar chart of top rules ranked by lift.
    """
    if rules_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No association rules available")
        return fig

    df = rules_df.head(top_n).copy()
    df = df.sort_values(by="lift", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df["lift"],
            y=df["rule_str"],
            orientation="h",
            marker=dict(
                color=df["lift"],
                colorscale="Viridis",
                showscale=False,
            ),
            text=[f"Lift: {l:.2f} | Conf: {c*100:.1f}%" for l, c in zip(df["lift"], df["confidence"])],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Lift: %{x:.2f}x<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"<b>Top {min(top_n, len(df))} Association Rules Ranked by Lift</b>",
        xaxis_title="Lift Ratio",
        yaxis_title="Association Rule (Antecedent -> Consequent)",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=max(350, len(df) * 26),
    )
    return fig


def plot_item_cooccurrence_heatmap(
    one_hot_df: pd.DataFrame, top_items: List[str]
) -> go.Figure:
    """
    Heatmap of item co-occurrence counts among the top N food items.
    """
    if one_hot_df.empty or not top_items:
        fig = go.Figure()
        fig.update_layout(title="No data available for heatmap")
        return fig

    valid_items = [item for item in top_items if item in one_hot_df.columns]
    if len(valid_items) < 2:
        fig = go.Figure()
        fig.update_layout(title="Insufficient items for co-occurrence heatmap")
        return fig

    sub_df = one_hot_df[valid_items].astype(int)
    # Compute dot product for co-occurrence matrix: X^T * X
    cooc_matrix = sub_df.T.dot(sub_df)

    fig = go.Figure(
        go.Heatmap(
            z=cooc_matrix.values,
            x=valid_items,
            y=valid_items,
            colorscale="YlOrRd",
            text=cooc_matrix.values,
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="Item 1: %{y}<br>Item 2: %{x}<br>Co-occurrences: %{z:,}<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Item Co-occurrence Frequency Matrix (Top Menu Items)</b>",
        xaxis_title="Menu Item",
        yaxis_title="Menu Item",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=520,
    )
    return fig


def plot_association_network_graph(
    rules_df: pd.DataFrame,
    max_rules: int = 35,
    min_lift: float = 1.0,
) -> go.Figure:
    """
    Interactive 2D Network Graph using NetworkX and Plotly.
    Nodes = Food items.
    Directed Edges = Association rules between items.
    Prevents browser freezing by restricting to the top strongest rules.
    """
    if rules_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No association rules available for network visualization")
        return fig

    filtered = rules_df[rules_df["lift"] >= min_lift].head(max_rules)
    if filtered.empty:
        fig = go.Figure()
        fig.update_layout(title="No rules qualify under current lift threshold")
        return fig

    # Build NetworkX DiGraph
    G = nx.DiGraph()

    for _, row in filtered.iterrows():
        antecedents = list(row["antecedents"])
        consequents = list(row["consequents"])
        lift_val = float(row["lift"])
        conf_val = float(row["confidence"])

        for ant in antecedents:
            G.add_node(ant, label=ant)
            for cons in consequents:
                G.add_node(cons, label=cons)
                G.add_edge(ant, cons, lift=lift_val, confidence=conf_val)

    if len(G.nodes) == 0:
        fig = go.Figure()
        fig.update_layout(title="Empty graph")
        return fig

    # Compute node layout positions using spring layout
    pos = nx.spring_layout(G, seed=42, k=0.6, iterations=50)

    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_text = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        lift = edge[2].get("lift", 1.0)
        conf = edge[2].get("confidence", 0.5)
        edge_text.append(f"{edge[0]} -> {edge[1]}<br>Lift: {lift:.2f}x<br>Conf: {conf*100:.1f}%")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.5, color="#FFAB91"),
        hoverinfo="none",
        mode="lines",
    )

    # Prepare node traces
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    node_labels = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        degree = G.degree(node)
        node_sizes.append(max(18, min(45, 15 + degree * 4)))
        node_labels.append(node)
        node_text.append(f"<b>{node}</b><br>Connected Rules: {degree}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="OrRd",
            color=[G.degree(n) for n in G.nodes()],
            size=node_sizes,
            colorbar=dict(title="Node Degree"),
            line=dict(color="#B71C1C", width=1.5),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="<b>Food Item Association Network Graph</b><br><i>Nodes = Menu Items | Edges = Discovered Association Rules</i>",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white",
            height=580,
        ),
    )
    return fig
