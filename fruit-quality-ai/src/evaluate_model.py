"""Evaluate the trained baseline fruit quality model on the test dataset.

This script:
1. Loads the saved Keras model
2. Loads test images from dataset/splits/test
3. Generates predictions for the full test set
4. Saves evaluation plots, reports, and a predictions CSV
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import MODELS_DIR, OUTPUTS_DIR, SPLITS_DATA_DIR


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MAX_MISCLASSIFIED_EXAMPLES = 9

MODEL_PATH = MODELS_DIR / "baseline_mobilenetv2.keras"
TEST_DIR = SPLITS_DATA_DIR / "test"

CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "figures" / "confusion_matrix.png"
NORMALISED_CONFUSION_MATRIX_PATH = (
    OUTPUTS_DIR / "figures" / "confusion_matrix_normalised.png"
)
MISCLASSIFIED_EXAMPLES_PATH = (
    OUTPUTS_DIR / "figures" / "misclassified_examples.png"
)
CLASSIFICATION_REPORT_PATH = OUTPUTS_DIR / "reports" / "classification_report.txt"
EVALUATION_SUMMARY_PATH = OUTPUTS_DIR / "reports" / "evaluation_summary.txt"
PREDICTIONS_CSV_PATH = OUTPUTS_DIR / "reports" / "test_predictions.csv"


def check_required_paths() -> None:
    """Stop early if the model file or test folder is missing."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model was not found: {MODEL_PATH}\n"
            "Run src/train_baseline.py before evaluation."
        )

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test dataset folder was not found: {TEST_DIR}\n"
            "Run src/split_dataset.py before evaluation."
        )


def load_test_dataset() -> tf.data.Dataset:
    """Load the test dataset without shuffling to preserve file order."""
    return tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )


def prepare_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Prefetch batches for smoother evaluation."""
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


def collect_true_labels(dataset: tf.data.Dataset) -> np.ndarray:
    """Collect all true labels from the batched dataset."""
    true_labels = []

    for _, labels in dataset:
        true_labels.extend(labels.numpy())

    return np.array(true_labels, dtype=int)


def create_predictions_dataframe(
    file_paths: list[str],
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    confidences: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    """Build a predictions table for CSV export."""
    rows = []

    for file_path, true_label, predicted_label, confidence in zip(
        file_paths,
        true_labels,
        predicted_labels,
        confidences,
    ):
        rows.append(
            {
                "image_path": file_path,
                "actual_class": class_names[int(true_label)],
                "predicted_class": class_names[int(predicted_label)],
                "confidence": float(confidence),
                "correct": bool(true_label == predicted_label),
            }
        )

    return pd.DataFrame(rows)


def save_confusion_matrix_image(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str,
    normalised: bool,
) -> None:
    """Save a confusion matrix heatmap image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 12))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    tick_positions = np.arange(len(class_names))
    plt.xticks(tick_positions, class_names, rotation=90)
    plt.yticks(tick_positions, class_names)

    value_format = ".2f" if normalised else "d"
    threshold = matrix.max() / 2 if matrix.size > 0 else 0

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            plt.text(
                column_index,
                row_index,
                format(value, value_format),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    plt.ylabel("Actual class")
    plt.xlabel("Predicted class")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_misclassified_examples(
    predictions_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a grid of a few misclassified image examples."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mistakes_df = predictions_df[~predictions_df["correct"]].head(MAX_MISCLASSIFIED_EXAMPLES)

    if mistakes_df.empty:
        plt.figure(figsize=(8, 3))
        plt.axis("off")
        plt.text(
            0.5,
            0.5,
            "No misclassified test images were found.",
            ha="center",
            va="center",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return

    columns = 3
    rows = int(np.ceil(len(mistakes_df) / columns))

    plt.figure(figsize=(15, 5 * rows))

    for plot_index, (_, row) in enumerate(mistakes_df.iterrows(), start=1):
        plt.subplot(rows, columns, plot_index)

        image = Image.open(row["image_path"]).convert("RGB")
        plt.imshow(image)
        plt.axis("off")
        plt.title(
            f"Actual: {row['actual_class']}\n"
            f"Pred: {row['predicted_class']}\n"
            f"Conf: {row['confidence']:.2f}",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_evaluation_summary(
    class_names: list[str],
    number_of_test_images: int,
    test_accuracy: float,
) -> str:
    """Build a short summary of the evaluation results."""
    lines = [
        "Model Evaluation Summary",
        "========================",
        f"Model path: {MODEL_PATH}",
        f"Test dataset path: {TEST_DIR}",
        f"Number of test images: {number_of_test_images}",
        f"Number of classes: {len(class_names)}",
        f"Class names: {', '.join(class_names)}",
        f"Overall test accuracy: {test_accuracy:.4f}",
        "",
        "Metric notes:",
        "Precision tells you how many predicted examples for a class were correct.",
        "Recall tells you how many real examples of a class the model found.",
        "F1-score is the balance between precision and recall.",
    ]

    return "\n".join(lines) + "\n"


def save_text_file(output_path: Path, text: str) -> None:
    """Save text content to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    """Run the full evaluation pipeline for the saved model."""
    check_required_paths()

    test_dataset = load_test_dataset()
    class_names = test_dataset.class_names
    file_paths = list(test_dataset.file_paths)

    true_labels = collect_true_labels(test_dataset)
    prepared_test_dataset = prepare_dataset(test_dataset)

    model = tf.keras.models.load_model(MODEL_PATH)

    prediction_scores = model.predict(prepared_test_dataset, verbose=1)
    predicted_labels = np.argmax(prediction_scores, axis=1)
    confidences = np.max(prediction_scores, axis=1)

    test_loss, _ = model.evaluate(prepared_test_dataset, verbose=0)
    test_accuracy = accuracy_score(true_labels, predicted_labels)

    predictions_df = create_predictions_dataframe(
        file_paths=file_paths,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        confidences=confidences,
        class_names=class_names,
    )
    PREDICTIONS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(PREDICTIONS_CSV_PATH, index=False)

    confusion = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
    )
    normalised_confusion = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        normalize="true",
    )

    save_confusion_matrix_image(
        matrix=confusion,
        class_names=class_names,
        output_path=CONFUSION_MATRIX_PATH,
        title="Confusion Matrix",
        normalised=False,
    )
    save_confusion_matrix_image(
        matrix=normalised_confusion,
        class_names=class_names,
        output_path=NORMALISED_CONFUSION_MATRIX_PATH,
        title="Normalised Confusion Matrix",
        normalised=True,
    )

    report_text = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    save_text_file(CLASSIFICATION_REPORT_PATH, report_text + "\n")

    summary_text = build_evaluation_summary(
        class_names=class_names,
        number_of_test_images=len(file_paths),
        test_accuracy=test_accuracy,
    )
    save_text_file(EVALUATION_SUMMARY_PATH, summary_text)

    save_misclassified_examples(
        predictions_df=predictions_df,
        output_path=MISCLASSIFIED_EXAMPLES_PATH,
    )

    print(summary_text)
    print(f"Classification report saved to: {CLASSIFICATION_REPORT_PATH}")
    print(f"Predictions CSV saved to: {PREDICTIONS_CSV_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    print(f"Normalised confusion matrix saved to: {NORMALISED_CONFUSION_MATRIX_PATH}")
    print(f"Misclassified examples saved to: {MISCLASSIFIED_EXAMPLES_PATH}")


if __name__ == "__main__":
    main()
