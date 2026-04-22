"""
Train YOLOv8n as a single-class traffic sign detector
Converts existing 250-class dataset to single-class ("sign") and trains YOLOv8n
"""

from ultralytics import YOLO
from pathlib import Path
import shutil
import yaml
import os


def convert_to_single_class(dataset_dir):
    """
    Convert multi-class YOLO labels to single class (all signs → class 0)
    Creates new label files, preserving originals
    """
    print("=" * 70)
    print("Converting to single-class labels")
    print("=" * 70)

    for split in ['train', 'val']:
        labels_dir = Path(dataset_dir) / split / 'labels'
        if not labels_dir.exists():
            print(f"  Skipping {split} (not found)")
            continue

        label_files = list(labels_dir.glob('*.txt'))
        print(f"  {split}: {len(label_files)} label files")

        for label_file in label_files:
            lines = label_file.read_text().strip().split('\n')
            new_lines = []
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    # Replace class ID with 0
                    parts[0] = '0'
                    new_lines.append(' '.join(parts))

            label_file.write_text('\n'.join(new_lines) + '\n')

    print("  Done!")


def create_single_class_yaml(dataset_dir):
    """Create data.yaml for single-class detection"""
    data_yaml = {
        'path': str(Path(dataset_dir).absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,
        'names': ['sign']
    }

    yaml_path = Path(dataset_dir) / 'data_single_class.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"  Created: {yaml_path}")
    return str(yaml_path)


def main():
    dataset_dir = 'data/output'
    project_dir = '/home/akshun-wsl/capstone-objectdet/data'
    name = 'single_class_detector'

    print("\nSingle-Class Traffic Sign Detector - YOLOv8n")
    print("=" * 70)

    # Check if train/val split exists
    if not (Path(dataset_dir) / 'train').exists():
        print("ERROR: No train/val split found. Run the training script first.")
        return

    # Backup original labels before converting
    for split in ['train', 'val']:
        labels_dir = Path(dataset_dir) / split / 'labels'
        backup_dir = Path(dataset_dir) / split / 'labels_multiclass_backup'
        if labels_dir.exists() and not backup_dir.exists():
            print(f"  Backing up {split} labels...")
            shutil.copytree(labels_dir, backup_dir)

    # Convert to single class
    convert_to_single_class(dataset_dir)

    # Create data.yaml
    print("\nCreating single-class data.yaml...")
    data_yaml_path = create_single_class_yaml(dataset_dir)

    # Train YOLOv8n
    print("\n" + "=" * 70)
    print("Training YOLOv8n Single-Class Detector")
    print("=" * 70)

    model = YOLO('yolov8n.pt')

    results = model.train(
        data=data_yaml_path,
        epochs=100,
        imgsz=640,
        batch=32,
        device='0',
        project=project_dir,
        name=name,
        exist_ok=True,

        # Single class = simpler task, can be more aggressive
        patience=30,
        save=True,
        save_period=10,

        # Augmentation
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.1,
        scale=0.4,
        fliplr=0.0,
        mosaic=0.5,     # Mosaic helps for single-class detection
        mixup=0.0,

        # Optimization
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,

        # Validation
        val=True,
        plots=True,
        verbose=True,
        workers=4,
    )

    best_model = Path(project_dir) / name / 'weights' / 'best.pt'
    print(f"\nTraining complete!")
    print(f"Best model: {best_model}")

    # Validate
    print("\nValidating...")
    model = YOLO(str(best_model))
    metrics = model.val()
    print(f"\n  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")


if __name__ == "__main__":
    try:
        from ultralytics import YOLO
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "ultralytics"])
        exit(0)

    main()
