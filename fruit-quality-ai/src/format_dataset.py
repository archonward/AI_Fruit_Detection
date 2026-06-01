"""Format the original fruit dataset into clean class folders.

This script reads images from:

dataset/raw/Original Image/

It writes cleaned copies to:

dataset/processed/

The raw dataset is never modified.
"""

import re
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from config import OUTPUTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR


SOURCE_DATA_DIR = RAW_DATA_DIR / "Original Image"
REPORT_PATH = OUTPUTS_DIR / "reports" / "dataset_formatting_report.txt"
VALID_EXTENSION = ".jpg"


def to_snake_case(class_name: str) -> str:
    """Convert names like FreshApple into fresh_apple."""
    words = re.findall(r"[A-Z][a-z]*|[a-z]+|[0-9]+", class_name)
    return "_".join(word.lower() for word in words)


def is_valid_image(image_path: Path) -> bool:
    """Return True if Pillow can open and verify the image."""
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def clear_processed_folder() -> None:
    """Safely clear only dataset/processed before formatting."""
    if PROCESSED_DATA_DIR.exists():
        shutil.rmtree(PROCESSED_DATA_DIR)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def format_class_folder(class_folder: Path) -> dict:
    """Copy valid JPG images from one raw class folder into dataset/processed."""
    processed_class_name = to_snake_case(class_folder.name)
    processed_class_dir = PROCESSED_DATA_DIR / processed_class_name
    processed_class_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    skipped_count = 0

    for image_path in sorted(class_folder.iterdir()):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() != VALID_EXTENSION:
            skipped_count += 1
            continue

        if not is_valid_image(image_path):
            skipped_count += 1
            continue

        copied_count += 1
        new_file_name = f"{processed_class_name}_{copied_count:06d}.jpg"
        destination_path = processed_class_dir / new_file_name

        shutil.copy2(image_path, destination_path)

    return {
        "original_class_name": class_folder.name,
        "processed_class_name": processed_class_name,
        "copied_count": copied_count,
        "skipped_count": skipped_count,
    }


def build_report(class_summaries: list[dict]) -> str:
    """Build a readable text report for the formatting step."""
    total_copied = sum(summary["copied_count"] for summary in class_summaries)
    total_skipped = sum(summary["skipped_count"] for summary in class_summaries)

    lines = [
        "Dataset Formatting Report",
        "=========================",
        f"Source folder: {SOURCE_DATA_DIR}",
        f"Processed folder: {PROCESSED_DATA_DIR}",
        "",
        f"Total classes formatted: {len(class_summaries)}",
        f"Total images copied: {total_copied}",
        f"Total skipped/corrupted images: {total_skipped}",
        "",
        "Class-by-class summary",
        "----------------------",
    ]

    for summary in class_summaries:
        lines.extend(
            [
                f"Original class: {summary['original_class_name']}",
                f"Processed class: {summary['processed_class_name']}",
                f"Images copied: {summary['copied_count']}",
                f"Skipped/corrupted images: {summary['skipped_count']}",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def save_report(report_text: str) -> None:
    """Save the formatting report to outputs/reports."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")


def main() -> None:
    """Format the raw dataset into clean processed class folders."""
    if not SOURCE_DATA_DIR.exists():
        report_text = (
            "Dataset Formatting Report\n"
            "=========================\n"
            f"Source folder was not found: {SOURCE_DATA_DIR}\n"
            "No files were copied.\n"
        )
        save_report(report_text)
        print(report_text)
        print(f"Report saved to: {REPORT_PATH}")
        return

    if not SOURCE_DATA_DIR.is_dir():
        report_text = (
            "Dataset Formatting Report\n"
            "=========================\n"
            f"Source path exists, but it is not a folder: {SOURCE_DATA_DIR}\n"
            "No files were copied.\n"
        )
        save_report(report_text)
        print(report_text)
        print(f"Report saved to: {REPORT_PATH}")
        return

    class_folders = [
        folder
        for folder in sorted(SOURCE_DATA_DIR.iterdir())
        if folder.is_dir()
    ]

    if not class_folders:
        report_text = (
            "Dataset Formatting Report\n"
            "=========================\n"
            f"No class folders were found inside: {SOURCE_DATA_DIR}\n"
            "No files were copied.\n"
        )
        save_report(report_text)
        print(report_text)
        print(f"Report saved to: {REPORT_PATH}")
        return

    clear_processed_folder()

    class_summaries = [
        format_class_folder(class_folder)
        for class_folder in class_folders
    ]

    report_text = build_report(class_summaries)
    save_report(report_text)

    print(report_text)
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
