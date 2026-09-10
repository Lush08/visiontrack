"""
VisionTrack - Dataset Utilities

Helpers for creating, validating, and converting datasets for YOLO26 training.
Supports YOLO format, COCO format, and Roboflow integration.
"""

from pathlib import Path
from typing import Optional

import yaml


# Default YOLO dataset YAML template
DATASET_TEMPLATE = """\
# VisionTrack Dataset Configuration
# Format: YOLO (Ultralytics compatible)

path: {dataset_path}   # Root directory
train: train/images    # Train images (relative to path)
val: val/images        # Val images (relative to path)
test: test/images      # Test images (optional, relative to path)

# Classes
names:
{class_names}
"""


def create_dataset_yaml(
    dataset_path: str,
    class_names: list[str],
    output_path: str = "data/custom.yaml",
) -> str:
    """
    Create a YOLO-format dataset YAML file.

    Args:
        dataset_path: Root path to the dataset directory
        class_names: List of class name strings
        output_path: Where to save the YAML file

    Returns:
        Path to created YAML file
    """
    class_names_str = "\n".join(
        f"  {i}: {name}" for i, name in enumerate(class_names)
    )

    content = DATASET_TEMPLATE.format(
        dataset_path=dataset_path,
        class_names=class_names_str,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)

    print(f"Dataset YAML created: {output_path}")
    print(f"  Classes ({len(class_names)}): {class_names[:5]}{'...' if len(class_names) > 5 else ''}")

    return str(output)


def validate_dataset_yaml(yaml_path: str) -> dict:
    """
    Validate a YOLO dataset YAML file.

    Checks:
    - File exists and is valid YAML
    - Required keys present (train, val, names)
    - Referenced directories exist
    - At least one class defined

    Returns:
        Parsed YAML content
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    # Check required keys
    required = ["train", "val", "names"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    # Check class names
    names = data["names"]
    if not names or len(names) == 0:
        raise ValueError("No class names defined")

    # Check directories (relative to dataset path)
    root = Path(data.get("path", path.parent))
    for split in ["train", "val"]:
        img_dir = root / data[split]
        if not img_dir.exists():
            print(f"  Warning: Directory not found: {img_dir}")

    print(f"Dataset validated: {yaml_path}")
    print(f"  Classes: {len(names)}")
    print(f"  Names: {list(names.values())[:5]}{'...' if len(names) > 5 else ''}")

    return data


def count_dataset_images(dataset_yaml: str) -> dict[str, int]:
    """Count images in each split of a YOLO dataset."""
    data = validate_dataset_yaml(dataset_yaml)
    root = Path(data.get("path", Path(dataset_yaml).parent))

    counts = {}
    for split in ["train", "val", "test"]:
        if split in data:
            img_dir = root / data[split].replace("/images", "/images")
            if img_dir.exists():
                images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
                counts[split] = len(images)
            else:
                counts[split] = 0
        else:
            counts[split] = 0

    total = sum(counts.values())
    print(f"Dataset counts: {counts} (total: {total})")
    return counts
