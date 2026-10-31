"""Exploratory analysis: summary tables printed to the console, figures saved to outputs/eda/.

The original notebook's interactive Plotly charts are kept in notebooks/; this
module reproduces the core analysis as static figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402

from .data import BINARY_COLS, CAT_COLS, MEMBERSHIP_ORDER, NUM_COLS, TARGET, missing_values_table, outlier_bounds  # noqa: E402

OUTPUT_DIR = Path("outputs/eda")
CONTINUOUS = NUM_COLS + ["age"]


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"Saved {path}")


def print_overview(df: pd.DataFrame) -> None:
    print(f"Rows: {len(df):,}  Columns: {df.shape[1]}  Duplicates: {df.duplicated().sum()}")
    print("\nMissing values:")
    print(missing_values_table(df).round(2).to_string())
    print("\nTarget balance:")
    print(df[TARGET].value_counts(normalize=True).round(3).to_string())
    print("\nSummary statistics:")
    print(df[CONTINUOUS].describe().round(2).to_string())
    print("\nSkewness:")
    print(df[CONTINUOUS].skew().round(3).to_string())
    print("\nKurtosis:")
    print(df[CONTINUOUS].kurtosis().round(3).to_string())
    print("\nIQR outlier fences (1.5 x IQR):")
    print(outlier_bounds(df, CONTINUOUS).round(2).to_string())
    print("\nImpossible values:")
    print(f"  negative points_in_wallet: {(df['points_in_wallet'] < 0).sum()}")
    print(f"  days_since_last_login == -999: {(df['days_since_last_login'] == -999).sum()}")
    print(f"  negative avg_time_spent: {(df['avg_time_spent'] < 0).sum()}")


def plot_distributions(df: pd.DataFrame) -> None:
    """Histogram with fitted normal curve, Q-Q plot and box plot for each continuous feature."""
    for col in CONTINUOUS:
        values = df[col].dropna()
        mu, sigma = stats.norm.fit(values)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(values, bins=50, density=True, color="teal", alpha=0.7)
        xs = pd.Series(sorted(values))
        axes[0].plot(xs, stats.norm.pdf(xs, mu, sigma), "k--", label="fitted normal")
        axes[0].set_title(f"Distribution of {col}")
        axes[0].legend()
        stats.probplot(values, dist="norm", plot=axes[1])
        axes[1].set_title(f"Q-Q plot of {col}")
        axes[2].boxplot(values, vert=False)
        axes[2].set_title(f"Box plot of {col}")
        _save(fig, f"distribution_{col}")


def plot_churn_by_category(df: pd.DataFrame) -> None:
    """Churn rate per level of every categorical and binary feature."""
    for col in CAT_COLS + BINARY_COLS:
        rates = df.groupby(col, observed=True)[TARGET].mean().mul(100)
        if col == "membership_category":
            rates = rates.reindex(MEMBERSHIP_ORDER)
        else:
            rates = rates.sort_values()
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(rates) + 1.5))
        rates.plot.barh(ax=ax, color="teal")
        ax.set_xlabel("Churn rate (%)")
        ax.set_ylabel("")
        ax.set_xlim(0, 100)
        ax.set_title(f"Churn rate by {col}")
        for i, v in enumerate(rates):
            ax.text(v + 1, i, f"{v:.0f}%", va="center")
        _save(fig, f"churn_by_{col}")


def plot_relationships(df: pd.DataFrame) -> None:
    """Scatter plots of key behaviour features coloured by churn, plus a correlation heatmap."""
    clean = df[
        (df["avg_time_spent"] > 0)
        & (df["avg_transaction_value"] > 0)
        & (df["points_in_wallet"] > 0)
        & (df["days_since_last_login"] > 0)
    ]
    pairs = [
        ("avg_time_spent", "avg_transaction_value"),
        ("avg_transaction_value", "points_in_wallet"),
        ("days_since_last_login", "avg_transaction_value"),
        ("age", "avg_transaction_value"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (x, y) in zip(axes.ravel(), pairs):
        sns.scatterplot(data=clean, x=x, y=y, hue=TARGET, palette=["lightgray", "teal"], s=8, alpha=0.6, ax=ax)
        ax.set_title(f"{x} vs {y}")
    _save(fig, "scatter_relationships")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[CONTINUOUS + [TARGET]].corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation between continuous features and churn")
    _save(fig, "correlation_heatmap")


def run(df: pd.DataFrame) -> None:
    print_overview(df)
    plot_distributions(df)
    plot_churn_by_category(df)
    plot_relationships(df)
