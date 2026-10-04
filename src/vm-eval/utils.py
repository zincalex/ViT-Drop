import json
from pathlib import Path


def check_dataset_formatting(dataset_name: str, base_dir: str, accelerator=None) -> bool:
    dataset_dir = Path(base_dir) / dataset_name
    imgs_dir = dataset_dir / "imgs"
    json_path = dataset_dir / f"{dataset_name}_demo.json"

    if not json_path.exists():
        accelerator.print(f"✗ Error: JSON file not found: {json_path}")
        accelerator.print(f"  Please create the JSON file manually with format:")
        accelerator.print(f'  [')
        accelerator.print(f'    {{"image": "imgs/image_0.png", "label": 0}},')
        accelerator.print(f'    {{"image": "imgs/image_1.png", "label": 1}},')
        accelerator.print(f'    ...')
        accelerator.print(f'  ]')
        return False
    accelerator.print(f"JSON file found: {json_path}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        accelerator.print(f"Error reading JSON file: {e}")
        return False

    invalid_entries = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            invalid_entries.append(f"Entry {i}: not a dictionary")
            continue
        if "image" not in entry: invalid_entries.append(f"Entry {i}: missing 'image' field")
        if "label" not in entry: invalid_entries.append(f"Entry {i}: missing 'label' field")
    if invalid_entries:
        accelerator.print(f"Error: Found {len(invalid_entries)} invalid entries:")
        for error in invalid_entries:
            accelerator.print(f"  - {error}")
        return False
    accelerator.print(f"All JSON entries have 'image' and 'label' fields")

    if not imgs_dir.exists():
        accelerator.print(f"Error: Images directory not found: {imgs_dir}")
        accelerator.print(f"  Please create the directory and add images: mkdir -p {imgs_dir}")
        return False
    accelerator.print(f"Images directory found: {imgs_dir}")

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}
    images = [img for img in imgs_dir.iterdir() if img.suffix.lower() in image_extensions]
    if not images:
        accelerator.print(f"Error: No images found in: {imgs_dir}")
        return False
    accelerator.print(f"Found {len(images)} images in imgs/ folder")

    return True
