"""Streamlit app for fruit quality classification.

This app loads the trained MobileNetV2 model and predicts the class of one
uploaded fruit image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st
import tensorflow as tf


# app/streamlit_app.py lives inside the app folder, so the project root is one
# level up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_mobilenetv2.keras"
TRAIN_DIR = PROJECT_ROOT / "dataset" / "splits" / "train"
IMAGE_SIZE = (224, 224)
TOP_K = 3
ORIGINAL_TEST_ACCURACY = "97.50%"
REAL_WORLD_ACCURACY = "81.82%"
EFFICIENTNET_REAL_WORLD_ACCURACY = "79.55%"
SELECTED_MODEL_NAME = "MobileNetV2"


def check_required_paths() -> None:
    """Stop early with clear messages when required files or folders are missing."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found: {MODEL_PATH}\n"
            "Make sure models/baseline_mobilenetv2.keras exists."
        )

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training class folder was not found: {TRAIN_DIR}\n"
            "Make sure dataset/splits/train/ exists."
        )


def load_class_names() -> list[str]:
    """Load class names in alphabetical folder order.

    This matches the default class ordering used by
    TensorFlow image_dataset_from_directory.
    """
    class_names = sorted(
        folder.name for folder in TRAIN_DIR.iterdir() if folder.is_dir()
    )

    if not class_names:
        raise FileNotFoundError(f"No class folders were found in: {TRAIN_DIR}")

    return class_names


@st.cache_resource
def load_model() -> tf.keras.Model:
    """Load the trained Keras model once and reuse it across reruns."""
    check_required_paths()
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_uploaded_image(uploaded_file) -> tuple[Image.Image, np.ndarray]:
    """Prepare the uploaded image exactly like src/predict_image.py.

    The corrected prediction script uses:
    - RGB images
    - resize to 224 x 224
    - raw pixel values in the 0-255 range
    - no MobileNetV2 preprocess_input call
    """
    try:
        preview_image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError as error:
        raise ValueError("The uploaded file is not a valid image.") from error
    except OSError as error:
        raise ValueError("The uploaded image could not be read.") from error

    resized_image = preview_image.resize(IMAGE_SIZE)

    # Keep the same pixel value range as the working single-image script.
    image_array = np.array(resized_image, dtype=np.float32)
    image_batch = np.expand_dims(image_array, axis=0)

    return preview_image, image_batch


def format_top_predictions(
    prediction_scores: np.ndarray, class_names: list[str], top_k: int
) -> list[tuple[str, float]]:
    """Return the top predicted classes with confidence scores."""
    top_indices = np.argsort(prediction_scores)[::-1][:top_k]
    return [(class_names[index], float(prediction_scores[index])) for index in top_indices]


def format_class_name(class_name: str) -> str:
    """Convert folder-style class names into readable labels."""
    return class_name.replace("_", " ").title()


def build_top_predictions_table(
    top_predictions: list[tuple[str, float]],
) -> pd.DataFrame:
    """Create a small table for the top prediction results."""
    return pd.DataFrame(
        [
            {
                "Rank": rank,
                "Class": format_class_name(class_name),
                "Confidence": f"{score * 100:.2f}%",
            }
            for rank, (class_name, score) in enumerate(top_predictions, start=1)
        ]
    )


def get_confidence_message(confidence: float) -> tuple[str, str]:
    """Return a short interpretation for the confidence score."""
    if confidence >= 0.80:
        return "success", "High confidence prediction"
    if confidence >= 0.50:
        return "info", "Moderate confidence prediction"
    return "warning", "Low confidence prediction - result may be unreliable"


def predict_image(uploaded_file) -> tuple[Image.Image, str, float, list[tuple[str, float]]]:
    """Run one prediction for the uploaded image."""
    preview_image, image_batch = preprocess_uploaded_image(uploaded_file)
    class_names = load_class_names()
    model = load_model()

    prediction_batch = model.predict(image_batch, verbose=0)
    prediction_scores = prediction_batch[0]

    predicted_index = int(np.argmax(prediction_scores))
    predicted_class = class_names[predicted_index]
    confidence = float(prediction_scores[predicted_index])
    top_predictions = format_top_predictions(prediction_scores, class_names, TOP_K)

    return preview_image, predicted_class, confidence, top_predictions


st.set_page_config(
    page_title="Fruit Quality Classification Demo",
    layout="centered",
)


st.title("Fruit Quality Classification Demo")
st.caption("Deployed model: MobileNetV2")
st.write(
    "This demo classifies uploaded fruit images into fresh or rotten fruit "
    "categories. The deployed model is MobileNetV2 because it performed best "
    "on the real-world phone-captured test set while remaining lightweight."
)
st.info(
    "For the clearest demo, upload one well-lit image with a single fruit in "
    "view. The app will show the predicted class, confidence score, and top "
    "three candidate labels."
)


st.sidebar.header("Project Results")
st.sidebar.metric("Selected model", SELECTED_MODEL_NAME)
st.sidebar.metric("Original test accuracy", ORIGINAL_TEST_ACCURACY)
st.sidebar.metric("Real-world accuracy", REAL_WORLD_ACCURACY)

st.sidebar.divider()
st.sidebar.header("Demo Summary")
st.sidebar.write("**Dataset:** Mendeley Fresh and Rotten Fruits Dataset")
st.sidebar.write("**Classes:** 16")
st.sidebar.write("**Input size:** 224 x 224")
st.sidebar.write("**Note:** CPU inference is supported")

with st.sidebar.expander("Model Comparison"):
    st.metric("MobileNetV2 real-world accuracy", REAL_WORLD_ACCURACY)
    st.metric("EfficientNetB0 real-world accuracy", EFFICIENTNET_REAL_WORLD_ACCURACY)
    st.write(
        "MobileNetV2 was selected because it performed slightly better on "
        "real-world images while remaining lightweight."
    )


uploaded_file = st.file_uploader(
    "Upload a fruit image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: .jpg, .jpeg, .png",
)


st.caption(
    "Supported image formats: JPG, JPEG, and PNG. The uploaded image is "
    "resized to 224 x 224 before inference."
)

st.info(
    "Suggested demo images: Try one clear single-fruit image first. Avoid "
    "multiple fruits in one image, and avoid very blurry or dark images."
)

st.subheader("Prediction Result")

if uploaded_file is None:
    st.write("Upload an image to see the predicted class and confidence scores.")
else:
    try:
        preview_image, predicted_class, confidence, top_predictions = predict_image(
            uploaded_file
        )
        confidence_level, confidence_message = get_confidence_message(confidence)
        top_predictions_table = build_top_predictions_table(top_predictions)

        image_column, result_column = st.columns([1.1, 1])

        with image_column:
            st.subheader("Uploaded Image")
            st.image(
                preview_image,
                caption=uploaded_file.name,
                use_container_width=True,
            )

        with result_column:
            st.subheader("Model Output")
            readable_prediction = format_class_name(predicted_class)
            st.success(f"Predicted class: {readable_prediction}")
            st.metric("Confidence", f"{confidence * 100:.2f}%")

            if confidence_level == "success":
                st.success(confidence_message)
            elif confidence_level == "info":
                st.info(confidence_message)
            else:
                st.warning(confidence_message)

            st.write("Top 3 predictions")
            st.table(top_predictions_table)

            with st.expander("How to interpret this result"):
                st.write(
                    "High confidence does not guarantee correctness. Real-world "
                    "lighting, background, camera angle, and fruit condition can "
                    "affect the model's prediction."
                )

    except FileNotFoundError as error:
        st.error(str(error))
    except ValueError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Prediction failed: {error}")


st.divider()
with st.expander("Limitations"):
    st.write(
        "- Trained on a controlled dataset.\n"
        "- The real-world dataset is still small.\n"
        "- Not designed for multiple fruits in one image.\n"
        "- Not a medical or commercial food safety tool.\n"
        "- Lighting, background, camera angle, and fruit condition can affect results."
    )
