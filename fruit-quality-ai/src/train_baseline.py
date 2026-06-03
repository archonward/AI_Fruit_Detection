"""Train a baseline fruit quality image classifier with TensorFlow/Keras.

This script expects the dataset to already be split like this:

dataset/splits/
├── train/
├── validation/
└── test/

Each split folder should contain the same class folders.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

from config import MODELS_DIR, OUTPUTS_DIR, SPLITS_DATA_DIR


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
RANDOM_SEED = 42

TRAIN_DIR = SPLITS_DATA_DIR / "train"
VALIDATION_DIR = SPLITS_DATA_DIR / "validation"
TEST_DIR = SPLITS_DATA_DIR / "test"

MODEL_PATH = MODELS_DIR / "baseline_mobilenetv2.keras"
ACCURACY_PLOT_PATH = OUTPUTS_DIR / "figures" / "baseline_accuracy.png"
LOSS_PLOT_PATH = OUTPUTS_DIR / "figures" / "baseline_loss.png"
REPORT_PATH = OUTPUTS_DIR / "reports" / "baseline_training_report.txt"


def check_split_folders() -> None:
    """Stop early with a clear message if a split folder is missing."""
    missing_folders = [
        split_dir
        for split_dir in [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]
        if not split_dir.exists()
    ]

    if missing_folders:
        missing_text = "\n".join(f"- {folder}" for folder in missing_folders)
        raise FileNotFoundError(
            "Missing dataset split folders:\n"
            f"{missing_text}\n\n"
            "Run src/split_dataset.py before training."
        )


def load_dataset(split_dir: Path, shuffle: bool) -> tf.data.Dataset:
    """Load one image split from class folders."""
    return tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=RANDOM_SEED,
    )


def prepare_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Cache and prefetch batches for smoother training."""
    return dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


def build_model(number_of_classes: int) -> tf.keras.Model:
    """Build a MobileNetV2 transfer learning model."""
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(number_of_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="baseline_mobilenetv2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_history_plot(
    history: tf.keras.callbacks.History,
    train_metric: str,
    validation_metric: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    """Save one training history line plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.history[train_metric]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history[train_metric], label=f"Train {y_label}")
    plt.plot(epochs, history.history[validation_metric], label=f"Validation {y_label}")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def build_report(
    class_names: list[str],
    history: tf.keras.callbacks.History,
    test_loss: float,
    test_accuracy: float,
) -> str:
    """Build the training report text."""
    final_train_accuracy = history.history["accuracy"][-1]
    final_validation_accuracy = history.history["val_accuracy"][-1]

    lines = [
        "Baseline MobileNetV2 Training Report",
        "====================================",
        f"Image size: {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}",
        f"Batch size: {BATCH_SIZE}",
        f"Number of classes: {len(class_names)}",
        f"Class names: {', '.join(class_names)}",
        f"Epochs: {EPOCHS}",
        "",
        f"Final train accuracy: {final_train_accuracy:.4f}",
        f"Final validation accuracy: {final_validation_accuracy:.4f}",
        f"Test accuracy: {test_accuracy:.4f}",
        f"Test loss: {test_loss:.4f}",
        "",
        f"Saved model: {MODEL_PATH}",
        f"Accuracy plot: {ACCURACY_PLOT_PATH}",
        f"Loss plot: {LOSS_PLOT_PATH}",
    ]

    return "\n".join(lines) + "\n"


def save_report(report_text: str) -> None:
    """Save the text report to outputs/reports."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")


def main() -> None:
    """Train, evaluate, and save the baseline model."""
    check_split_folders()

    train_dataset = load_dataset(TRAIN_DIR, shuffle=True)
    validation_dataset = load_dataset(VALIDATION_DIR, shuffle=False)
    test_dataset = load_dataset(TEST_DIR, shuffle=False)

    class_names = train_dataset.class_names
    number_of_classes = len(class_names)

    train_dataset = prepare_dataset(train_dataset)
    validation_dataset = prepare_dataset(validation_dataset)
    test_dataset = prepare_dataset(test_dataset)

    model = build_model(number_of_classes)

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
    )

    test_loss, test_accuracy = model.evaluate(test_dataset)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    save_history_plot(
        history,
        train_metric="accuracy",
        validation_metric="val_accuracy",
        title="Baseline Training Accuracy",
        y_label="Accuracy",
        output_path=ACCURACY_PLOT_PATH,
    )
    save_history_plot(
        history,
        train_metric="loss",
        validation_metric="val_loss",
        title="Baseline Training Loss",
        y_label="Loss",
        output_path=LOSS_PLOT_PATH,
    )

    report_text = build_report(
        class_names=class_names,
        history=history,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
    )
    save_report(report_text)

    print(report_text)
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
