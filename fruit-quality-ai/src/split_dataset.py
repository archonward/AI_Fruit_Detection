"""Split the processed fruit dataset into train, validation, and test folders.

This script reads from:

dataset/processed/

It writes copied images to:

dataset/splits/train/
dataset/splits/validation/
dataset/splits/test/

The raw and processed datasets are never deleted or modified.
"""

import random
import shutil
from pathlib import Path

from config import OUTPUTS_DIR, PROCESSED_DATA_DIR, SPLITS_DATA_DIR


RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

TRAIN_DIR = SPLITS_DATA_DIR / "train"
VALIDATION_DIR = SPLITS_DATA_DIR / "validation"
TEST_DIR = SPLITS_DATA_DIR / "test"
REPORT_PATH = OUTPUTS_DIR / "reports" / "dataset_split_report.txt"

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clear_split_folders() -> None:
    """Clear only the split output folders so the script is safe to rerun."""
    for split_dir in [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]:
        if split_dir.exists():
            shutil.rmtree(split_dir)

        split_dir.mkdir(parents=True, exist_ok=True)


def list_image_files(class_folder: Path) -> list[Path]:
    """Return valid image files directly inside one processed class folder."""
    return [
        file_path
        for file_path in sorted(class_folder.iterdir())
        if file_path.is_file()
        and file_path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]


def split_images(image_paths: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    """Shuffle images and split them into train, validation, and test lists."""
    shuffled_images = image_paths.copy()
    random.Random(RANDOM_SEED).shuffle(shuffled_images)

    total_images = len(shuffled_images)
    train_count = int(total_images * TRAIN_RATIO)
    validation_count = int(total_images * VALIDATION_RATIO)

    train_images = shuffled_images[:train_count]
    validation_images = shuffled_images[train_count:train_count + validation_count]
    test_images = shuffled_images[train_count + validation_count:]

    return train_images, validation_images, test_images


def copy_images(image_paths: list[Path], destination_class_dir: Path) -> None:
    """Copy images into one split class folder."""
    destination_class_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        destination_path = destination_class_dir / image_path.name
        shutil.copy2(image_path, destination_path)


def split_class_folder(class_folder: Path) -> dict:
    """Split and copy one processed class folder."""
    image_paths = list_image_files(class_folder)
    train_images, validation_images, test_images = split_images(image_paths)

    copy_images(train_images, TRAIN_DIR / class_folder.name)
    copy_images(validation_images, VALIDATION_DIR / class_folder.name)
    copy_images(test_images, TEST_DIR / class_folder.name)

    return {
        "class_name": class_folder.name,
        "total_images": len(image_paths),
        "train_count": len(train_images),
        "validation_count": len(validation_images),
        "test_count": len(test_images),
    }


def build_report(class_summaries: list[dict]) -> str:
    """Build a readable report for the split step."""
    total_images = sum(summary["total_images"] for summary in class_summaries)
    total_train = sum(summary["train_count"] for summary in class_summaries)
    total_validation = sum(summary["validation_count"] for summary in class_summaries)
    total_test = sum(summary["test_count"] for summary in class_summaries)

    lines = [
        "Dataset Split Report",
        "====================",
        f"Processed source folder: {PROCESSED_DATA_DIR}",
        f"Splits output folder: {SPLITS_DATA_DIR}",
        f"Random seed: {RANDOM_SEED}",
        f"Split ratio: {int(TRAIN_RATIO * 100)}% train, "
        f"{int(VALIDATION_RATIO * 100)}% validation, "
        f"{100 - int(TRAIN_RATIO * 100) - int(VALIDATION_RATIO * 100)}% test",
        "",
        f"Total classes split: {len(class_summaries)}",
        f"Total images: {total_images}",
        f"Total train images: {total_train}",
        f"Total validation images: {total_validation}",
        f"Total test images: {total_test}",
        "",
        "Class-by-class summary",
        "----------------------",
    ]

    for summary in class_summaries:
        lines.extend(
            [
                f"Class: {summary['class_name']}",
                f"  Total images: {summary['total_images']}",
                f"  Train: {summary['train_count']}",
                f"  Validation: {summary['validation_count']}",
                f"  Test: {summary['test_count']}",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def save_report(report_text: str) -> None:
    """Save the split report to outputs/reports."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")


def main() -> None:
    """Create train, validation, and test split folders."""
    if not PROCESSED_DATA_DIR.exists():
        report_text = (
            "Dataset Split Report\n"
            "====================\n"
            f"Processed dataset folder was not found: {PROCESSED_DATA_DIR}\n"
            "No split folders were created.\n"
        )
        save_report(report_text)
        print(report_text)
        print(f"Report saved to: {REPORT_PATH}")
        return

    if not PROCESSED_DATA_DIR.is_dir():
        report_text = (
            "Dataset Split Report\n"
            "====================\n"
            f"Processed dataset path exists, but it is not a folder: {PROCESSED_DATA_DIR}\n"
            "No split folders were created.\n"
        )
        save_report(report_text)
        print(report_text)
        print(f"Report saved to: {REPORT_PATH}")
        return

    class_folders = [
        folder
        for folder in sorted(PROCESSED_DATA_DIR.iterdir())
        if folder.is_dir()
    ]

    if not class_folders:
        report_text = (
            "Dataset Split Report\n"
            "====================\n"
            f"No class folders were found inside: {PROCESSED_DATA_DIR}\n"
            "No split folders were created.\n"
        )
        save_report(report_text)
        print(report_text)
        print(f"Report saved to: {REPORT_PATH}")
        return

    clear_split_folders()

    class_summaries = [
        split_class_folder(class_folder)
        for class_folder in class_folders
    ]

    report_text = build_report(class_summaries)
    save_report(report_text)

    print(report_text)
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
