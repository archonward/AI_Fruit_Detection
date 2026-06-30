"""Evaluate the EfficientNetB0 model on the real-world fruit image dataset.

This script:
1. Loads the saved EfficientNetB0 Keras model
2. Loads class names from dataset/splits/train in alphabetical folder order
3. Reads real-world images from dataset/real_world
4. Applies the same preprocessing used by src/train_efficientnet.py
5. Saves CSV and text reports
6. Optionally saves a grid of misclassified examples

Expected local command:
    python src/evaluate_real_world_efficientnet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from config import DATASET_DIR, MODELS_DIR, OUTPUTS_DIR, SPLITS_DATA_DIR


IMAGE_SIZE = (224, 224)
MAX_MISCLASSIFIED_EXAMPLES = 9
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_PATH = MODELS_DIR / "efficientnetb0.keras"
TRAIN_DIR = SPLITS_DATA_DIR / "train"
REAL_WORLD_DIR = DATASET_DIR / "real_world"

PREDICTIONS_CSV_PATH = (
    OUTPUTS_DIR / "reports" / "efficientnetb0_real_world_predictions.csv"
)
REPORT_PATH = (
    OUTPUTS_DIR / "reports" / "efficientnetb0_real_world_evaluation_report.txt"
)
MISCLASSIFIED_EXAMPLES_PATH = (
    OUTPUTS_DIR
    / "figures"
    / "efficientnetb0_real_world_misclassified_examples.png"
)


def check_required_paths() -> None:
    """Stop early with clear messages when required files or folders are missing."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"EfficientNetB0 model file was not found: {MODEL_PATH}\n"
            "Run src/train_efficientnet.py before evaluation."
        )

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training class folder was not found: {TRAIN_DIR}\n"
            "Run src/split_dataset.py before evaluation."
        )

    if not REAL_WORLD_DIR.exists():
        raise FileNotFoundError(
            f"Real-world dataset folder was not found: {REAL_WORLD_DIR}\n"
            "Create dataset/real_world/ before running this script."
        )


def load_class_names() -> list[str]:
    """Load class folder names in alphabetical order.

    This matches the ordering used by image_dataset_from_directory during
    training, which is important because prediction indices map to class names.
    """
    class_names = sorted(
        folder.name for folder in TRAIN_DIR.iterdir() if folder.is_dir()
    )

    if not class_names:
        raise FileNotFoundError(f"No class folders were found in: {TRAIN_DIR}")

    return class_names


def list_real_world_class_folders() -> list[Path]:
    """Return the real-world class folders in alphabetical order."""
    class_folders = sorted(folder for folder in REAL_WORLD_DIR.iterdir() if folder.is_dir())

    if not class_folders:
        raise FileNotFoundError(f"No class folders were found in: {REAL_WORLD_DIR}")

    return class_folders


def list_image_files(class_folder: Path) -> list[Path]:
    """Return supported image files from one class folder."""
    return [
        file_path
        for file_path in sorted(class_folder.iterdir())
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def load_and_preprocess_image(image_path: Path) -> np.ndarray:
    """Prepare one image for EfficientNetB0 inference.

    train_efficientnet.py feeds raw resized RGB image tensors into the model.
    The EfficientNet application model handles its own rescaling internally, so
    this function should not apply MobileNetV2 preprocessing or manual
    EfficientNet preprocessing.
    """
    try:
        image = tf.keras.utils.load_img(
            image_path,
            color_mode="rgb",
            target_size=IMAGE_SIZE,
        )
        image_array = tf.keras.utils.img_to_array(image)
    except UnidentifiedImageError as error:
        raise ValueError(f"Unsupported or unreadable image file: {image_path}") from error
    except OSError as error:
        raise ValueError(f"Could not read image file: {image_path}") from error

    return np.expand_dims(image_array, axis=0)


def predict_one_image(
    model: tf.keras.Model,
    image_path: Path,
    class_names: list[str],
) -> tuple[str, float]:
    """Predict one image and return the class name and confidence."""
    image_batch = load_and_preprocess_image(image_path)
    prediction_batch = model.predict(image_batch, verbose=0)
    prediction_scores = prediction_batch[0]

    predicted_index = int(np.argmax(prediction_scores))
    predicted_class = class_names[predicted_index]
    confidence = float(prediction_scores[predicted_index])

    return predicted_class, confidence


def evaluate_real_world_dataset(
    model: tf.keras.Model,
    model_class_names: list[str],
) -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    """Evaluate every valid real-world image and collect warnings."""
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    per_class_accuracy: dict[str, float] = {}

    for class_folder in list_real_world_class_folders():
        actual_class = class_folder.name

        if actual_class not in model_class_names:
            warnings.append(
                "Skipped class folder because it is not part of the model class list: "
                f"{class_folder}"
            )
            continue

        image_paths = list_image_files(class_folder)

        if not image_paths:
            warnings.append(
                "Skipped empty class folder or folder with no supported images: "
                f"{class_folder}"
            )
            continue

        class_rows_before = len(rows)

        for image_path in image_paths:
            try:
                predicted_class, confidence = predict_one_image(
                    model=model,
                    image_path=image_path,
                    class_names=model_class_names,
                )
            except ValueError as error:
                warnings.append(str(error))
                continue
            except Exception as error:
                warnings.append(f"Prediction failed for {image_path}: {error}")
                continue

            rows.append(
                {
                    "image_path": str(image_path),
                    "actual_class": actual_class,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "correct": bool(actual_class == predicted_class),
                }
            )

        class_rows_after = rows[class_rows_before:]
        if class_rows_after:
            correct_count = sum(1 for row in class_rows_after if row["correct"])
            per_class_accuracy[actual_class] = correct_count / len(class_rows_after)
        else:
            warnings.append(
                "No readable images could be evaluated for class folder: "
                f"{class_folder}"
            )

    predictions_df = pd.DataFrame(
        rows,
        columns=[
            "image_path",
            "actual_class",
            "predicted_class",
            "confidence",
            "correct",
        ],
    )

    if predictions_df.empty:
        raise ValueError(
            "No valid real-world images were evaluated. Check the dataset folders, "
            "image formats, and image readability."
        )

    return predictions_df, per_class_accuracy, warnings


def build_summary_report(
    predictions_df: pd.DataFrame,
    per_class_accuracy: dict[str, float],
    warnings: list[str],
    misclassification_note: str,
) -> str:
    """Build the text report for the real-world evaluation step."""
    total_images = len(predictions_df)
    correct_count = int(predictions_df["correct"].sum())
    overall_accuracy = float(predictions_df["correct"].mean())

    lines = [
        "EfficientNetB0 Real-World Evaluation Report",
        "===========================================",
        f"Model path: {MODEL_PATH}",
        f"Training class folder: {TRAIN_DIR}",
        f"Real-world dataset path: {REAL_WORLD_DIR}",
        f"Image size: {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}",
        f"Total images evaluated: {total_images}",
        f"Number correct: {correct_count}",
        f"Overall accuracy: {overall_accuracy:.4f}",
        "",
        "Per-class accuracy:",
    ]

    if per_class_accuracy:
        for class_name in sorted(per_class_accuracy):
            lines.append(f"- {class_name}: {per_class_accuracy[class_name]:.4f}")
    else:
        lines.append("- No per-class accuracy could be calculated.")

    lines.extend(
        [
            "",
            "Preprocessing note:",
            "Images were converted to RGB, resized to 224 x 224, converted to",
            "NumPy arrays, and passed to the saved model without manual",
            "MobileNetV2 or EfficientNet preprocessing.",
            "",
            f"Misclassified examples note: {misclassification_note}",
        ]
    )

    if warnings:
        lines.extend(["", "Warnings and skipped items:"])
        for warning in warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines) + "\n"


def save_text_file(output_path: Path, text: str) -> None:
    """Save text content to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def save_misclassified_examples(predictions_df: pd.DataFrame) -> str:
    """Save a grid of misclassified images when any are present."""
    mistakes_df = predictions_df[~predictions_df["correct"]].head(MAX_MISCLASSIFIED_EXAMPLES)

    if mistakes_df.empty:
        return "No misclassifications were found, so no image grid was saved."

    MISCLASSIFIED_EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)

    columns = 3
    rows = int(np.ceil(len(mistakes_df) / columns))

    plt.figure(figsize=(15, 5 * rows))

    for plot_index, (_, row) in enumerate(mistakes_df.iterrows(), start=1):
        plt.subplot(rows, columns, plot_index)

        try:
            image = Image.open(row["image_path"]).convert("RGB")
            plt.imshow(image)
            plt.axis("off")
            plt.title(
                f"Actual: {row['actual_class']}\n"
                f"Pred: {row['predicted_class']}\n"
                f"Conf: {row['confidence']:.2f}",
                fontsize=10,
            )
        except OSError:
            plt.axis("off")
            plt.text(
                0.5,
                0.5,
                "Could not reopen image",
                ha="center",
                va="center",
                fontsize=10,
            )
            plt.title(
                f"Actual: {row['actual_class']}\n"
                f"Pred: {row['predicted_class']}",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(MISCLASSIFIED_EXAMPLES_PATH, dpi=200)
    plt.close()

    return f"Saved misclassified image grid to: {MISCLASSIFIED_EXAMPLES_PATH}"


def print_summary(predictions_df: pd.DataFrame, per_class_accuracy: dict[str, float]) -> None:
    """Print a short console summary."""
    total_images = len(predictions_df)
    correct_count = int(predictions_df["correct"].sum())
    overall_accuracy = float(predictions_df["correct"].mean())

    print("EfficientNetB0 Real-World Evaluation Summary")
    print("============================================")
    print(f"Total images evaluated: {total_images}")
    print(f"Number correct: {correct_count}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print("Per-class accuracy:")

    if per_class_accuracy:
        for class_name in sorted(per_class_accuracy):
            print(f"  - {class_name}: {per_class_accuracy[class_name]:.4f}")
    else:
        print("  - No per-class accuracy could be calculated.")


def main() -> int:
    """Run the EfficientNetB0 real-world evaluation pipeline."""
    try:
        check_required_paths()
        model_class_names = load_class_names()
        model = tf.keras.models.load_model(MODEL_PATH)

        predictions_df, per_class_accuracy, warnings = evaluate_real_world_dataset(
            model=model,
            model_class_names=model_class_names,
        )

        PREDICTIONS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(PREDICTIONS_CSV_PATH, index=False)

        misclassification_note = save_misclassified_examples(predictions_df)
        report_text = build_summary_report(
            predictions_df=predictions_df,
            per_class_accuracy=per_class_accuracy,
            warnings=warnings,
            misclassification_note=misclassification_note,
        )
        save_text_file(REPORT_PATH, report_text)

        print_summary(predictions_df, per_class_accuracy)
        print(f"Predictions CSV saved to: {PREDICTIONS_CSV_PATH}")
        print(f"Report saved to: {REPORT_PATH}")
        print(misclassification_note)
        if warnings:
            print("Warnings:")
            for warning in warnings:
                print(f"  - {warning}")

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
