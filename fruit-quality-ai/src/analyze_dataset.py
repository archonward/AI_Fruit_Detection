"""Inspect the raw fruit image dataset.

This script expects the dataset to be organized like this:

dataset/raw/
├── fresh_apples/
├── rotten_apples/
└── ...

Each class should have its own folder inside dataset/raw.
"""

from pathlib import Path

from config import RAW_DATA_DIR


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(class_folder: Path) -> int:
    """Count image files inside one class folder."""
    image_count = 0

    for file_path in class_folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_count += 1

    return image_count


def main() -> None:
    """Print a simple summary of class folders and image counts."""
    if not RAW_DATA_DIR.exists():
        print(f"Raw dataset folder was not found: {RAW_DATA_DIR}")
        print("Add the dataset later inside dataset/raw, then run this script again.")
        return

    class_folders = [
        folder
        for folder in RAW_DATA_DIR.iterdir()
        if folder.is_dir()
    ]

    if not class_folders:
        print(f"No class folders were found inside: {RAW_DATA_DIR}")
        print("Expected one folder per class, for example fresh_apples or rotten_apples.")
        return

    total_images = 0

    print("Dataset summary")
    print("===============")

    for class_folder in sorted(class_folders):
        image_count = count_images(class_folder)
        total_images += image_count
        print(f"{class_folder.name}: {image_count} images")

    print()
    print(f"Total classes: {len(class_folders)}")
    print(f"Total images: {total_images}")


if __name__ == "__main__":
    main()
