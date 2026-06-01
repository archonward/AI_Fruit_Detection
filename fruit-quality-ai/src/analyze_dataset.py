"""Inspect the original raw fruit image dataset.

This script only reads files from:

dataset/raw/Original Image/

It does not rename, move, split, process, augment, or train anything.
"""

from pathlib import Path
from statistics import median

from PIL import Image, UnidentifiedImageError

from config import OUTPUTS_DIR, RAW_DATA_DIR


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ORIGINAL_IMAGE_DIR = RAW_DATA_DIR / "Original Image"
REPORT_PATH = OUTPUTS_DIR / "reports" / "dataset_inspection_report.txt"


def is_readable_image(image_path: Path) -> bool:
    """Return True if Pillow can open and verify the image."""
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def inspect_class_folder(class_folder: Path) -> dict:
    """Count valid, skipped, corrupted, and extension details for one class."""
    valid_images = 0
    skipped_non_images = 0
    corrupted_images = []
    extensions_used = set()

    for file_path in sorted(class_folder.rglob("*")):
        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()

        if extension not in VALID_IMAGE_EXTENSIONS:
            skipped_non_images += 1
            continue

        extensions_used.add(extension)

        if is_readable_image(file_path):
            valid_images += 1
        else:
            corrupted_images.append(file_path)

    return {
        "class_name": class_folder.name,
        "valid_images": valid_images,
        "skipped_non_images": skipped_non_images,
        "corrupted_images": corrupted_images,
        "extensions_used": extensions_used,
    }


def find_low_image_warnings(class_summaries: list[dict]) -> list[str]:
    """Warn when a class has much fewer images than the typical class."""
    image_counts = [summary["valid_images"] for summary in class_summaries]

    if len(image_counts) < 2:
        return []

    typical_count = median(image_counts)

    if typical_count == 0:
        return []

    warning_limit = typical_count * 0.5
    warnings = []

    for summary in class_summaries:
        image_count = summary["valid_images"]
        if image_count < warning_limit:
            warnings.append(
                f"Warning: {summary['class_name']} has only {image_count} valid images. "
                f"This is much fewer than the typical class count of {typical_count:.0f}."
            )

    return warnings


def build_report() -> str:
    """Build the dataset inspection report text."""
    lines = [
        "Dataset Inspection Report",
        "=========================",
        f"Inspected folder: {ORIGINAL_IMAGE_DIR}",
        "",
    ]

    if not ORIGINAL_IMAGE_DIR.exists():
        lines.append("Status: Missing folder.")
        lines.append("Create dataset/raw/Original Image before running this script.")
        return "\n".join(lines)

    if not ORIGINAL_IMAGE_DIR.is_dir():
        lines.append("Status: Path exists, but it is not a folder.")
        return "\n".join(lines)

    class_folders = [
        folder
        for folder in sorted(ORIGINAL_IMAGE_DIR.iterdir())
        if folder.is_dir()
    ]

    if not class_folders:
        lines.append("Status: No class folders were found.")
        lines.append("Expected one folder per class inside dataset/raw/Original Image.")
        return "\n".join(lines)

    class_summaries = [
        inspect_class_folder(class_folder)
        for class_folder in class_folders
    ]

    total_valid_images = sum(summary["valid_images"] for summary in class_summaries)
    total_skipped_non_images = sum(
        summary["skipped_non_images"] for summary in class_summaries
    )
    total_corrupted_images = sum(
        len(summary["corrupted_images"]) for summary in class_summaries
    )
    all_extensions = sorted(
        extension
        for summary in class_summaries
        for extension in summary["extensions_used"]
    )

    lines.extend(
        [
            f"Total classes: {len(class_summaries)}",
            f"Total valid images: {total_valid_images}",
            f"Skipped non-image files: {total_skipped_non_images}",
            f"Corrupted or unreadable images: {total_corrupted_images}",
            f"Image extensions used: {', '.join(sorted(set(all_extensions))) or 'None'}",
            "",
            "Class-by-class summary",
            "----------------------",
        ]
    )

    for summary in class_summaries:
        extensions = ", ".join(sorted(summary["extensions_used"])) or "None"
        lines.extend(
            [
                f"Class: {summary['class_name']}",
                f"  Valid images: {summary['valid_images']}",
                f"  Skipped non-image files: {summary['skipped_non_images']}",
                f"  Corrupted or unreadable images: {len(summary['corrupted_images'])}",
                f"  Extensions used: {extensions}",
            ]
        )

        if summary["corrupted_images"]:
            lines.append("  Corrupted files:")
            for image_path in summary["corrupted_images"]:
                lines.append(f"    - {image_path}")

        lines.append("")

    warnings = find_low_image_warnings(class_summaries)

    if warnings:
        lines.append("Warnings")
        lines.append("--------")
        lines.extend(warnings)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    """Print the report and save it to outputs/reports."""
    report_text = build_report()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
