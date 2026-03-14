from __future__ import annotations

"""
Batch prediction CLI for churn model artifacts.

Usage example:
python src/predict_cli.py --input-csv data/WA_Fn-UseC_-Telco-Customer-Churn.csv --output-csv models/predictions.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run batch churn predictions from CSV.")
    parser.add_argument("--input-csv", required=True, help="Path to input CSV.")
    parser.add_argument("--output-csv", required=True, help="Path to write predictions CSV.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional custom threshold. Defaults to metadata threshold.",
    )
    return parser.parse_args()



def load_artifacts(project_root: Path):
    """Load model + metadata created by training script."""
    model_path = project_root / "models" / "churn_model.joblib"
    metadata_path = project_root / "models" / "model_metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run src/train_churn_model.py first."
        )

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata



def align_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    numeric_features: list[str],
) -> pd.DataFrame:
    """Align input dataframe with model feature schema."""
    aligned = df.copy()
    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = None

    aligned = aligned[feature_columns]
    aligned = aligned.replace(" ", None)

    for column in numeric_features:
        if column in aligned.columns:
            aligned[column] = pd.to_numeric(aligned[column], errors="coerce")

    return aligned



def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    model, metadata = load_artifacts(project_root)
    threshold = float(args.threshold) if args.threshold is not None else float(metadata["threshold"])

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_csv(input_path)
    features = align_features(
        source_df,
        metadata["feature_columns"],
        metadata.get("numeric_features", []),
    )

    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    result_df = source_df.copy()
    result_df["churn_probability"] = probabilities
    result_df["predicted_churn"] = predictions

    result_df.to_csv(output_path, index=False)

    print("Prediction completed.")
    print(f"Input rows: {len(result_df)}")
    print(f"Threshold used: {threshold:.2f}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
