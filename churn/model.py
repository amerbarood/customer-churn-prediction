"""Preprocessing pipeline, decision tree training, tuning and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn.metrics as skmet  # noqa: E402
from scipy.stats import randint  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline, make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from .data import BINARY_COLS, MEMBERSHIP_ORDER, NUM_COLS, TARGET  # noqa: E402
from .transformers import MakeOrdinal, OutlierTrans, ReplaceVals  # noqa: E402

# Feature groups used by the model (as in the notebook's modelling section).
MODEL_NUM_COLS = NUM_COLS
MODEL_AGE_COLS = ["age_group"]
MODEL_MEMBERSHIP_COLS = ["membership_category"]
MODEL_CAT_COLS = ["gender", "medium_of_operation", "internet_option", "feedback", "complaint_status"]
MODEL_BINARY_COLS = BINARY_COLS
FEATURE_COLS = MODEL_NUM_COLS + MODEL_BINARY_COLS + MODEL_CAT_COLS + MODEL_AGE_COLS + MODEL_MEMBERSHIP_COLS

MEMBERSHIP_MAP = {"membership_category": {name: i for i, name in enumerate(MEMBERSHIP_ORDER)}}

DEFAULT_MODEL_PATH = Path("models/churn_tree.joblib")
OUTPUT_DIR = Path("outputs")


@dataclass
class Metrics:
    accuracy: float
    sensitivity: float
    specificity: float
    precision: float

    def __str__(self) -> str:
        return (
            f"accuracy: {self.accuracy:.2f}, sensitivity: {self.sensitivity:.2f}, "
            f"specificity: {self.specificity:.2f}, precision: {self.precision:.2f}"
        )


def split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 0):
    return train_test_split(df[FEATURE_COLS], df[TARGET], test_size=test_size, random_state=random_state)


def build_preprocessor() -> ColumnTransformer:
    def ohe():
        return OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

    return ColumnTransformer(
        transformers=[
            ("stand", make_pipeline(OutlierTrans(3), SimpleImputer(strategy="median"), StandardScaler()), MODEL_NUM_COLS),
            ("spec1", make_pipeline(MakeOrdinal(), StandardScaler()), MODEL_AGE_COLS),
            ("spec2", make_pipeline(ReplaceVals(MEMBERSHIP_MAP), StandardScaler()), MODEL_MEMBERSHIP_COLS),
            ("bin", make_pipeline(ohe()), MODEL_BINARY_COLS),
            ("cat", make_pipeline(ohe()), MODEL_CAT_COLS),
        ]
    )


def build_pipeline(**tree_params) -> Pipeline:
    return make_pipeline(build_preprocessor(), DecisionTreeClassifier(**tree_params))


def evaluate(model, X_test, y_test) -> Metrics:
    pred = model.predict(X_test)
    y = np.asarray(y_test).ravel()
    return Metrics(
        accuracy=skmet.accuracy_score(y, pred),
        sensitivity=skmet.recall_score(y, pred),
        specificity=skmet.recall_score(y, pred, pos_label=0),
        precision=skmet.precision_score(y, pred),
    )


def feature_names(pipe: Pipeline) -> np.ndarray:
    coltrans = pipe.named_steps["columntransformer"]
    bin_cols = coltrans.named_transformers_["bin"].named_steps["onehotencoder"].get_feature_names_out(MODEL_BINARY_COLS)
    cat_cols = coltrans.named_transformers_["cat"].named_steps["onehotencoder"].get_feature_names_out(MODEL_CAT_COLS)
    return np.concatenate([MODEL_NUM_COLS + MODEL_AGE_COLS + MODEL_MEMBERSHIP_COLS, bin_cols, cat_cols])


def feature_importances(pipe: Pipeline) -> pd.Series:
    tree = pipe.named_steps["decisiontreeclassifier"]
    imp = pd.Series(tree.tree_.compute_feature_importances(normalize=False), index=feature_names(pipe))
    return imp[imp > 0].sort_values(ascending=False)


def tune(X_train, y_train, n_iter: int = 20, cv: int = 5, random_state: int = 0) -> RandomizedSearchCV:
    params = {
        "decisiontreeclassifier__min_samples_leaf": randint(100, 1200),
        "decisiontreeclassifier__max_depth": randint(2, 11),
    }
    search = RandomizedSearchCV(
        build_pipeline(random_state=0), params, cv=cv, n_iter=n_iter, scoring="roc_auc", random_state=random_state
    )
    search.fit(X_train, np.asarray(y_train).ravel())
    return search


def plot_confusion_matrix(model, X_test, y_test, name: str = "confusion_matrix") -> Path:
    cm = skmet.confusion_matrix(np.asarray(y_test).ravel(), model.predict(X_test))
    display = skmet.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    display.plot()
    display.ax_.set(title="Churn Prediction Confusion Matrix", xlabel="Predicted Value", ylabel="Actual Value")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.png"
    display.figure_.savefig(path, bbox_inches="tight")
    plt.close(display.figure_)
    print(f"Saved {path}")
    return path


def run(df: pd.DataFrame, n_iter: int = 20, model_path: Path | None = DEFAULT_MODEL_PATH) -> dict[str, Metrics]:
    X_train, X_test, y_train, y_test = split(df)
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    print("\n== Baseline decision tree (max_depth=2, min_samples_leaf=5) ==")
    baseline = build_pipeline(min_samples_leaf=5, max_depth=2)
    baseline.fit(X_train, y_train)
    results = {"baseline": evaluate(baseline, X_test, y_test)}
    print(results["baseline"])
    print("Features used by the tree:")
    print(feature_importances(baseline).to_string())

    print(f"\n== Randomized search ({n_iter} candidates, 5-fold CV, ROC AUC) ==")
    search = tune(X_train, y_train, n_iter=n_iter)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV ROC AUC: {search.best_score_:.4f}")
    results["tuned"] = evaluate(search.best_estimator_, X_test, y_test)
    print(results["tuned"])
    plot_confusion_matrix(search.best_estimator_, X_test, y_test)

    if model_path is not None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(search.best_estimator_, model_path)
        print(f"Saved model to {model_path}")
    return results


def predict(model_path: Path, df: pd.DataFrame) -> np.ndarray:
    model = joblib.load(model_path)
    return model.predict(df[FEATURE_COLS])
