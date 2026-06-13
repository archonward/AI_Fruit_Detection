"""Analyse mistakes from the saved real-world evaluation predictions CSV."""

from __future__ import annotations

import sys

import pandas as pd

from config import OUTPUTS_DIR


PREDICTIONS_CSV_PATH = OUTPUTS_DIR / "reports" / "real_world_predictions.csv"
MISCLASSIFIED_CSV_PATH = OUTPUTS_DIR / "reports" / "real_world_misclassified_only.csv"
REPORT_PATH = OUTPUTS_DIR / "reports" / "real_world_error_analysis.txt"

REQUIRED_COLUMNS = [
    "image_path",
    "actual_class",
    "predicted_class",
    "confidence",
    "correct",
]


def check_required_file() -> None:
    """Stop early if the predictions CSV is missing."""
    if not PREDICTIONS_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Predictions CSV was not found: {PREDICTIONS_CSV_PATH}\n"
            "Run src/evaluate_real_world.py before this analysis step."
        )


def load_predictions() -> pd.DataFrame:
    """Load and validate the saved real-world predictions CSV."""
    predictions_df = pd.read_csv(PREDICTIONS_CSV_PATH)

    missing_columns = [
        column_name
        for column_name in REQUIRED_COLUMNS
        if column_name not in predictions_df.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(
            "Predictions CSV is missing required columns: "
            f"{missing_text}"
        )

    predictions_df = predictions_df[REQUIRED_COLUMNS].copy()

    # Convert the correct column to real booleans in case it was saved as text.
    predictions_df["correct"] = (
        predictions_df["correct"].astype(str).str.strip().str.lower() == "true"
    )

    if predictions_df.empty:
        raise ValueError("Predictions CSV is empty.")

    return predictions_df


def build_mistakes_per_actual_class(misclassified_df: pd.DataFrame) -> pd.Series:
    """Count the number of mistakes for each actual class."""
    return misclassified_df["actual_class"].value_counts().sort_index()


def build_confusion_counts(misclassified_df: pd.DataFrame) -> pd.Series:
    """Count actual-to-predicted confusions for misclassified examples only."""
    return (
        misclassified_df.groupby(["actual_class", "predicted_class"])
        .size()
        .sort_values(ascending=False)
    )


def build_report(
    predictions_df: pd.DataFrame,
    misclassified_df: pd.DataFrame,
    mistakes_per_actual_class: pd.Series,
    confusion_counts: pd.Series,
) -> str:
    """Build the text report for real-world error analysis."""
    total_predictions = len(predictions_df)
    total_correct = int(predictions_df["correct"].sum())
    total_incorrect = len(misclassified_df)
    overall_accuracy = total_correct / total_predictions

    lines = [
        "Real-World Error Analysis",
        "=========================",
        f"Predictions CSV path: {PREDICTIONS_CSV_PATH}",
        f"Total predictions: {total_predictions}",
        f"Total correct: {total_correct}",
        f"Total incorrect: {total_incorrect}",
        f"Overall accuracy: {overall_accuracy:.4f}",
        "",
        "Mistakes per actual class:",
    ]

    if mistakes_per_actual_class.empty:
        lines.append("- No mistakes were found.")
    else:
        for actual_class, count in mistakes_per_actual_class.items():
            lines.append(f"- {actual_class}: {int(count)}")

    lines.extend(["", "Actual class to predicted class confusion counts:"])

    if confusion_counts.empty:
        lines.append("- No confusions were found.")
    else:
        for (actual_class, predicted_class), count in confusion_counts.items():
            lines.append(
                f"- {actual_class} -> {predicted_class}: {int(count)}"
            )

    return "\n".join(lines) + "\n"


def save_text_file(report_text: str) -> None:
    """Save the text report to outputs/reports."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")


def print_summary(
    predictions_df: pd.DataFrame,
    misclassified_df: pd.DataFrame,
    mistakes_per_actual_class: pd.Series,
    confusion_counts: pd.Series,
) -> None:
    """Print the error analysis summary to the console."""
    total_predictions = len(predictions_df)
    total_correct = int(predictions_df["correct"].sum())
    total_incorrect = len(misclassified_df)
    overall_accuracy = total_correct / total_predictions

    print("Real-World Error Analysis")
    print("=========================")
    print(f"Total predictions: {total_predictions}")
    print(f"Total correct: {total_correct}")
    print(f"Total incorrect: {total_incorrect}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print("Mistakes per actual class:")

    if mistakes_per_actual_class.empty:
        print("  - No mistakes were found.")
    else:
        for actual_class, count in mistakes_per_actual_class.items():
            print(f"  - {actual_class}: {int(count)}")

    print("Actual class to predicted class confusion counts:")
    if confusion_counts.empty:
        print("  - No confusions were found.")
    else:
        for (actual_class, predicted_class), count in confusion_counts.items():
            print(f"  - {actual_class} -> {predicted_class}: {int(count)}")


def main() -> int:
    """Run the real-world error analysis pipeline."""
    try:
        check_required_file()
        predictions_df = load_predictions()
        misclassified_df = predictions_df[~predictions_df["correct"]].copy()

        mistakes_per_actual_class = build_mistakes_per_actual_class(misclassified_df)
        confusion_counts = build_confusion_counts(misclassified_df)

        MISCLASSIFIED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        misclassified_df.to_csv(MISCLASSIFIED_CSV_PATH, index=False)

        report_text = build_report(
            predictions_df=predictions_df,
            misclassified_df=misclassified_df,
            mistakes_per_actual_class=mistakes_per_actual_class,
            confusion_counts=confusion_counts,
        )
        save_text_file(report_text)

        print_summary(
            predictions_df=predictions_df,
            misclassified_df=misclassified_df,
            mistakes_per_actual_class=mistakes_per_actual_class,
            confusion_counts=confusion_counts,
        )
        print(f"Misclassified-only CSV saved to: {MISCLASSIFIED_CSV_PATH}")
        print(f"Report saved to: {REPORT_PATH}")

        return 0

    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Unexpected error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
