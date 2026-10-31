"""Loading and cleaning the churn dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA = Path("data/churn.csv")

TARGET = "churn_risk_score"

NUM_COLS = [
    "days_since_last_login",
    "avg_time_spent",
    "avg_transaction_value",
    "points_in_wallet",
    "avg_frequency_login_days",
]
CAT_COLS = [
    "gender",
    "region_category",
    "preferred_offer_types",
    "medium_of_operation",
    "internet_option",
    "feedback",
    "complaint_status",
    "membership_category",
]
BINARY_COLS = [
    "joined_through_referral",
    "used_special_discount",
    "offer_application_preference",
    "past_complaint",
]
UNUSED_COLS = ["security_no", "referral_id", "joining_date", "last_visit_time", "Unnamed: 0"]

MEMBERSHIP_ORDER = [
    "No Membership",
    "Basic Membership",
    "Silver Membership",
    "Gold Membership",
    "Premium Membership",
    "Platinum Membership",
]

AGE_BINS = [10, 19, 29, 39, 49, 59, 120]
AGE_LABELS = ["10-19", "20-29", "30-39", "40-49", "50-59", "60+"]


def load_raw(path: Path = DEFAULT_DATA) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the data-quality issues found during exploration."""
    df = df.copy()
    df["avg_frequency_login_days"] = df["avg_frequency_login_days"].replace("Error", np.nan)
    df["gender"] = df["gender"].replace("Unknown", "Unspecified")
    df["joined_through_referral"] = df["joined_through_referral"].replace("?", "Unspecified")
    df["medium_of_operation"] = df["medium_of_operation"].replace("?", "Unspecified")
    df[NUM_COLS] = df[NUM_COLS].astype(float)
    df["age"] = df["age"].astype(int)
    df[TARGET] = df[TARGET].astype(int)
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    return df


def load(path: Path = DEFAULT_DATA) -> pd.DataFrame:
    return clean(load_raw(path))


def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    table = pd.DataFrame({"missing": missing, "percent": 100 * missing / len(df)})
    return table[table["missing"] > 0].sort_values("percent", ascending=False)


def outlier_bounds(df: pd.DataFrame, cols: list[str], multiplier: float = 1.5) -> pd.DataFrame:
    """IQR fences for each column."""
    rows = {}
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = multiplier * (q3 - q1)
        rows[col] = {"lower": q1 - iqr, "upper": q3 + iqr}
    return pd.DataFrame(rows).T
