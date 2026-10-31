"""Command-line entry point.

Examples:
    python main.py eda                 # summary tables + figures in outputs/eda/
    python main.py train               # baseline + tuned decision tree, saves models/churn_tree.joblib
    python main.py train --n-iter 50   # wider hyperparameter search
    python main.py predict --input data/churn.csv --output outputs/predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from churn import data, eda, model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Customer churn analysis and prediction")
    parser.add_argument("--data", type=Path, default=data.DEFAULT_DATA, help="path to churn.csv")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("eda", help="exploratory analysis: tables and figures")

    p = sub.add_parser("train", help="train, tune and evaluate the decision tree")
    p.add_argument("--n-iter", type=int, default=20, help="randomized search candidates")
    p.add_argument("--model", type=Path, default=model.DEFAULT_MODEL_PATH)

    p = sub.add_parser("predict", help="predict churn for customers in a CSV")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/predictions.csv"))
    p.add_argument("--model", type=Path, default=model.DEFAULT_MODEL_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "eda":
        eda.run(data.load(args.data))
    elif args.command == "train":
        model.run(data.load(args.data), n_iter=args.n_iter, model_path=args.model)
    elif args.command == "predict":
        df = data.load(args.input)
        df["predicted_churn"] = model.predict(args.model, df)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Predicted churn for {len(df)} customers ({df['predicted_churn'].mean():.0%} at risk) -> {args.output}")


if __name__ == "__main__":
    main()
