"""Plotting utilities for optimization results."""

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def plot_optimization_trajectory(
    results: List[Dict[str, Any]],
    title: str = "Optimization Trajectory",
) -> go.Figure:
    """Plot score vs iteration with best-so-far line.

    Args:
        results: List of dicts with 'score' and 'params' keys
        title: Plot title

    Returns:
        Plotly figure
    """
    scores = [r["score"] for r in results]
    iterations = list(range(1, len(scores) + 1))

    # Calculate best-so-far
    best_so_far = []
    current_best = float("-inf")
    for s in scores:
        current_best = max(current_best, s)
        best_so_far.append(current_best)

    fig = go.Figure()

    # Individual scores
    fig.add_trace(go.Scatter(
        x=iterations,
        y=scores,
        mode="markers+lines",
        name="Score",
        line=dict(color="steelblue", width=1),
        marker=dict(size=8, color="steelblue"),
    ))

    # Best so far
    fig.add_trace(go.Scatter(
        x=iterations,
        y=best_so_far,
        mode="lines",
        name="Best So Far",
        line=dict(color="crimson", width=2, dash="dash"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title="Score (Target Phase Fraction)",
        template="plotly_white",
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
    )

    return fig


def plot_parameter_exploration(
    results: List[Dict[str, Any]],
    param_name: str,
    title: Optional[str] = None,
) -> go.Figure:
    """Plot score vs a single parameter.

    Args:
        results: List of dicts with 'score' and 'params' keys
        param_name: Name of the parameter to plot
        title: Optional plot title

    Returns:
        Plotly figure
    """
    scores = [r["score"] for r in results]
    values = [r["params"].get(param_name) for r in results]
    iterations = list(range(1, len(scores) + 1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=values,
        y=scores,
        mode="markers",
        marker=dict(
            size=10,
            color=iterations,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Iteration"),
        ),
        text=[f"Iter {i}" for i in iterations],
        hovertemplate=f"{param_name}: %{{x}}<br>Score: %{{y:.4f}}<br>%{{text}}<extra></extra>",
    ))

    fig.update_layout(
        title=title or f"Score vs {param_name}",
        xaxis_title=param_name,
        yaxis_title="Score",
        template="plotly_white",
    )

    return fig


def plot_parameter_grid(
    results: List[Dict[str, Any]],
    continuous_params: Optional[List[str]] = None,
) -> go.Figure:
    """Plot parameter exploration in a grid layout.

    Args:
        results: List of dicts with 'score' and 'params' keys
        continuous_params: List of continuous parameter names to plot.
            If None, auto-detects numeric parameters.

    Returns:
        Plotly figure with subplots
    """
    if not results:
        return go.Figure()

    # Auto-detect continuous params
    if continuous_params is None:
        continuous_params = []
        for key, value in results[0]["params"].items():
            if isinstance(value, (int, float)):
                continuous_params.append(key)

    n_params = len(continuous_params)
    if n_params == 0:
        return go.Figure()

    # Create subplot grid
    n_cols = min(2, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=continuous_params,
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    scores = [r["score"] for r in results]
    iterations = list(range(1, len(scores) + 1))

    for i, param in enumerate(continuous_params):
        row = i // n_cols + 1
        col = i % n_cols + 1

        values = [r["params"].get(param) for r in results]

        fig.add_trace(
            go.Scatter(
                x=values,
                y=scores,
                mode="markers",
                marker=dict(
                    size=8,
                    color=iterations,
                    colorscale="Viridis",
                    showscale=(i == 0),  # Only show colorbar once
                    colorbar=dict(title="Iter") if i == 0 else None,
                ),
                name=param,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(title_text=param, row=row, col=col)
        fig.update_yaxes(title_text="Score" if col == 1 else "", row=row, col=col)

    fig.update_layout(
        title="Parameter Exploration",
        template="plotly_white",
        height=300 * n_rows,
    )

    return fig


def plot_categorical_comparison(
    results: List[Dict[str, Any]],
    param_name: str,
    title: Optional[str] = None,
) -> go.Figure:
    """Plot score distribution for each categorical value.

    Args:
        results: List of dicts with 'score' and 'params' keys
        param_name: Name of the categorical parameter
        title: Optional plot title

    Returns:
        Plotly figure with box plots
    """
    df = pd.DataFrame([
        {"category": r["params"].get(param_name), "score": r["score"]}
        for r in results
    ])

    categories = df["category"].unique()

    fig = go.Figure()

    for cat in sorted(categories):
        cat_scores = df[df["category"] == cat]["score"]
        fig.add_trace(go.Box(
            y=cat_scores,
            name=str(cat),
            boxmean=True,
        ))

    fig.update_layout(
        title=title or f"Score by {param_name}",
        xaxis_title=param_name,
        yaxis_title="Score",
        template="plotly_white",
        showlegend=False,
    )

    return fig


def plot_optimization_summary(
    results: List[Dict[str, Any]],
    target_phase: str = "Target Phase",
) -> go.Figure:
    """Create a comprehensive summary plot with trajectory and top results.

    Args:
        results: List of dicts with 'score' and 'params' keys
        target_phase: Name of the target phase for labeling

    Returns:
        Plotly figure with multiple subplots
    """
    # Get top 5 results
    sorted_results = sorted(results, key=lambda r: r["score"], reverse=True)
    top_5 = sorted_results[:5]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=[
            f"Optimization Trajectory ({target_phase})",
            "Top 5 Configurations",
        ],
        row_heights=[0.6, 0.4],
        vertical_spacing=0.15,
    )

    # Trajectory plot
    scores = [r["score"] for r in results]
    iterations = list(range(1, len(scores) + 1))

    best_so_far = []
    current_best = float("-inf")
    for s in scores:
        current_best = max(current_best, s)
        best_so_far.append(current_best)

    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=scores,
            mode="markers+lines",
            name="Score",
            line=dict(color="steelblue", width=1),
            marker=dict(size=6, color="steelblue"),
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=best_so_far,
            mode="lines",
            name="Best So Far",
            line=dict(color="crimson", width=2, dash="dash"),
        ),
        row=1, col=1,
    )

    # Top 5 bar chart
    labels = [f"#{i+1}" for i in range(len(top_5))]
    top_scores = [r["score"] for r in top_5]

    # Create hover text with params
    hover_texts = []
    for r in top_5:
        params = r["params"]
        text = "<br>".join(f"{k}: {v}" for k, v in params.items())
        hover_texts.append(text)

    fig.add_trace(
        go.Bar(
            x=labels,
            y=top_scores,
            marker_color="steelblue",
            text=[f"{s:.3f}" for s in top_scores],
            textposition="outside",
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=False,
        ),
        row=2, col=1,
    )

    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_xaxes(title_text="Rank", row=2, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=600,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def load_results_from_json(filepath: str) -> List[Dict[str, Any]]:
    """Load optimization results from a JSON file.

    Args:
        filepath: Path to JSON file with 'all_results' key

    Returns:
        List of result dicts
    """
    import json
    with open(filepath, "r") as f:
        data = json.load(f)
    return data.get("all_results", [])
