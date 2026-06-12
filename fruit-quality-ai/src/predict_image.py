"""Predict the class of one fruit image with the trained baseline model.

Example:
    python src/predict_image.py dataset/splits/test/fresh_apple/fresh_apple_000001.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import UnidentifiedImageError

from config import MODELS_DIR, SPLITS_DATA_DIR


IMAGE_SIZE = (224, 224)
MODEL_PATH = MODELS_DIR / "baseline_mobilenetv2.keras"
TRAIN_DIR = SPLITS_DATA_DIR / "train"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
TOP_K = 3


def parse_arguments() -> argparse.Namespace:
    """Read the image path from the command line."""
    parser = argparse.ArgumentParser(
        description="Predict the fruit quality class for one image."
    )
    parser.add_argument("image_path", help="Path to the image file to classify.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print input tensor shape, dtype, and pixel value range.",
    )
    return parser.parse_args()


def check_required_paths(image_path: Path) -> None:
    """Stop early with clear messages when required files or folders are missing."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found: {MODEL_PATH}\n"
            "Run src/train_baseline.py before using this script."
        )

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training class folder was not found: {TRAIN_DIR}\n"
            "Run src/split_dataset.py before using this script."
        )

    if not image_path.exists():
        raise FileNotFoundError(f"Image file was not found: {image_path}")

    if not image_path.is_file():
        raise FileNotFoundError(f"Image path is not a file: {image_path}")

    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported image format: {image_path.suffix or '[no extension]'}\n"
            f"Supported formats: {supported}"
        )


def load_class_names() -> list[str]:
    """Load class folder names in alphabetical order.

    This matches the default ordering used by image_dataset_from_directory.
    """
    class_names = sorted(
        folder.name for folder in TRAIN_DIR.iterdir() if folder.is_dir()
    )

    if not class_names:
        raise FileNotFoundError(f"No class folders were found in: {TRAIN_DIR}")

    return class_names


def load_and_preprocess_image(image_path: Path) -> np.ndarray:
    """Open one image and prepare it exactly like the evaluation pipeline.

    train_baseline.py applies MobileNetV2 preprocessing inside the model.
    evaluate_model.py then feeds raw resized RGB float32 image tensors to the
    saved model. To match that behavior, this function must not call
    mobilenet_v2.preprocess_input manually.
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

    # image_dataset_from_directory returns float32 image tensors in the 0-255
    # range before the model performs its internal preprocessing.
    return np.expand_dims(image_array, axis=0)


def format_top_predictions(
    prediction_scores: np.ndarray, class_names: list[str], top_k: int
) -> list[tuple[str, float]]:
    """Return the top predicted classes with confidence scores."""
    top_indices = np.argsort(prediction_scores)[::-1][:top_k]
    return [(class_names[index], float(prediction_scores[index])) for index in top_indices]


def print_debug_info(image_batch: np.ndarray) -> None:
    """Print input tensor details for debugging preprocessing issues."""
    print(f"Input image shape: {image_batch.shape}")
    print(f"Input image dtype: {image_batch.dtype}")
    print(f"Min pixel value: {float(np.min(image_batch)):.4f}")
    print(f"Max pixel value: {float(np.max(image_batch)):.4f}")


def main() -> int:
    """Load the model, predict one image, and print the results."""
    args = parse_arguments()
    image_path = Path(args.image_path).resolve()

    try:
        check_required_paths(image_path)
        class_names = load_class_names()
        image_batch = load_and_preprocess_image(image_path)

        model = tf.keras.models.load_model(MODEL_PATH)
        prediction_batch = model.predict(image_batch, verbose=0)
        prediction_scores = prediction_batch[0]

        predicted_index = int(np.argmax(prediction_scores))
        predicted_class = class_names[predicted_index]
        confidence = float(prediction_scores[predicted_index])
        top_predictions = format_top_predictions(prediction_scores, class_names, TOP_K)

        print(f"Image path: {image_path}")
        print(f"Class names: {class_names}")
        if args.debug:
            print_debug_info(image_batch)
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence score: {confidence:.4f}")
        print("Top 3 predictions:")
        for rank, (class_name, score) in enumerate(top_predictions, start=1):
            print(f"  {rank}. {class_name}: {score:.4f}")

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
