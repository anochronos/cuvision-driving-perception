"""
MobileNetV3-Small Traffic Sign Classifier
Lightweight classifier for Stage 2 of the two-stage detection pipeline
Trains on transparent PNG sign images with augmentation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import random
import numpy as np
import json
import time


class SignPNGDataset(Dataset):
    """
    Dataset that loads transparent PNG signs and augments them
    to simulate real-world cropped sign appearances
    """

    def __init__(self, signs_dir, backgrounds_dir=None, samples_per_class=200, img_size=96,
                 include_negative=True, negative_multiplier=3,
                 extra_class_dirs=None):
        self.signs_dir = Path(signs_dir)
        self.img_size = img_size

        # Load sign files
        self.sign_files = sorted(list(self.signs_dir.glob('*.png')))
        if not self.sign_files:
            raise ValueError(f"No PNGs found in {signs_dir}")

        # Class mapping -sign classes first, then extra classes, then not_a_sign last
        self.class_names = [f.stem for f in self.sign_files]

        # Load extra class directories (e.g. traffic lights)
        # Each entry: {'name': 'traffic_light', 'dir': 'path/to/crops'}
        self.extra_classes = {}
        if extra_class_dirs:
            for entry in extra_class_dirs:
                name = entry['name']
                crop_dir = Path(entry['dir'])
                crops = sorted([f for f in crop_dir.glob('*.jpg')] +
                               [f for f in crop_dir.glob('*.png')])
                if crops:
                    class_id = len(self.class_names)
                    self.class_names.append(name)
                    self.extra_classes[class_id] = crops
                    print(f"  Extra class '{name}': {len(crops)} images (id={class_id})")

        if include_negative:
            self.class_names.append('not_a_sign')
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        self.not_a_sign_id = self.class_to_id.get('not_a_sign', -1)

        # Load background patches for compositing
        self.bg_patches = []
        if backgrounds_dir and Path(backgrounds_dir).exists():
            bg_dir = Path(backgrounds_dir)
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                for bg_file in bg_dir.glob(ext):
                    try:
                        img = Image.open(bg_file).convert('RGB')
                        self.bg_patches.append(img)
                    except Exception:
                        continue

        num_sign_classes = len(self.sign_files)
        num_extra = len(self.extra_classes)
        print(f"Loaded {num_sign_classes} sign classes, {num_extra} extra classes, "
              f"{len(self.bg_patches)} backgrounds"
              f"{', + not_a_sign class' if include_negative else ''}")

        # Pre-generate sample indices (class_idx, augmentation_seed)
        self.samples = []
        for class_idx in range(num_sign_classes):
            for i in range(samples_per_class):
                self.samples.append((class_idx, random.randint(0, 999999)))

        # Add extra class samples (use actual crop count, augmented up to samples_per_class)
        for class_id, crops in self.extra_classes.items():
            num_samples = max(samples_per_class, len(crops))
            for i in range(num_samples):
                self.samples.append((class_id, random.randint(0, 999999)))

        # Add negative samples (random background crops -no sign)
        if include_negative:
            neg_samples = samples_per_class * negative_multiplier
            for i in range(neg_samples):
                self.samples.append((self.not_a_sign_id, random.randint(0, 999999)))
            print(f"  Added {neg_samples} negative (not_a_sign) samples")

        # Downscale→upscale augmentation to simulate low-res distant crops
        class RandomDownscaleUpscale:
            """Shrink image to tiny size then scale back up to simulate real-world low-res crops"""
            def __init__(self, min_size=8, max_size=32, probability=0.6):
                self.min_size = min_size
                self.max_size = max_size
                self.probability = probability

            def __call__(self, img):
                if random.random() < self.probability:
                    orig_size = img.size
                    tiny = random.randint(self.min_size, self.max_size)
                    # Downscale with random interpolation (simulates different capture quality)
                    down_method = random.choice([Image.BILINEAR, Image.NEAREST, Image.BOX])
                    img = img.resize((tiny, tiny), down_method)
                    # Upscale back with bilinear (simulates what pipeline does)
                    img = img.resize(orig_size, Image.BILINEAR)
                return img

        # Augmentation transforms applied AFTER compositing
        self.post_transform = transforms.Compose([
            RandomDownscaleUpscale(min_size=8, max_size=32, probability=0.6),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.01),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.is_val = False

    def __len__(self):
        return len(self.samples)

    def _composite_sign(self, sign_img, seed):
        """Composite a sign onto a background patch with augmentation"""
        rng = random.Random(seed)

        # Convert sign to RGBA
        sign = sign_img.convert('RGBA')

        # Random scale (simulate different crop sizes)
        scale = rng.uniform(0.5, 0.95)
        new_size = int(self.img_size * scale)
        if new_size < 16:
            new_size = 16
        sign = sign.resize((new_size, new_size), Image.LANCZOS)

        # Random rotation
        angle = rng.uniform(-15, 15)
        sign = sign.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))

        # Create background
        if self.bg_patches and rng.random() < 0.7:
            bg = rng.choice(self.bg_patches).copy()
            # Random crop from background
            bw, bh = bg.size
            if bw > self.img_size and bh > self.img_size:
                x = rng.randint(0, bw - self.img_size)
                y = rng.randint(0, bh - self.img_size)
                bg = bg.crop((x, y, x + self.img_size, y + self.img_size))
            else:
                bg = bg.resize((self.img_size, self.img_size))
        else:
            # Random solid/noisy background
            r, g, b = rng.randint(50, 200), rng.randint(50, 200), rng.randint(50, 200)
            bg = Image.new('RGB', (self.img_size, self.img_size), (r, g, b))

        # Center the sign on the background with slight offset
        sw, sh = sign.size
        max_offset_x = max(1, (self.img_size - sw) // 4)
        max_offset_y = max(1, (self.img_size - sh) // 4)
        offset_x = (self.img_size - sw) // 2 + rng.randint(-max_offset_x, max_offset_x)
        offset_y = (self.img_size - sh) // 2 + rng.randint(-max_offset_y, max_offset_y)

        # Clamp offsets
        offset_x = max(0, min(offset_x, self.img_size - sw))
        offset_y = max(0, min(offset_y, self.img_size - sh))

        # Paste sign onto background using alpha channel
        bg.paste(sign, (offset_x, offset_y), sign)

        return bg

    def _random_background_crop(self, seed):
        """Generate a random background crop (no sign) for negative samples"""
        rng = random.Random(seed)

        if self.bg_patches:
            bg = rng.choice(self.bg_patches).copy()
            bw, bh = bg.size

            # Random crop at various scales to simulate different false positive sizes
            crop_scale = rng.uniform(0.05, 0.4)
            crop_size = max(32, int(min(bw, bh) * crop_scale))

            if bw > crop_size and bh > crop_size:
                x = rng.randint(0, bw - crop_size)
                y = rng.randint(0, bh - crop_size)
                bg = bg.crop((x, y, x + crop_size, y + crop_size))

            bg = bg.resize((self.img_size, self.img_size), Image.LANCZOS)
        else:
            # Random noise image
            arr = np.random.RandomState(seed).randint(0, 255,
                    (self.img_size, self.img_size, 3), dtype=np.uint8)
            bg = Image.fromarray(arr)

        return bg

    def __getitem__(self, idx):
        class_idx, seed = self.samples[idx]
        rng = random.Random(seed)

        if class_idx == self.not_a_sign_id:
            # Negative sample -random background crop
            img = self._random_background_crop(seed)
        elif class_idx in self.extra_classes:
            # Extra class (e.g. traffic light) -load real crop
            crops = self.extra_classes[class_idx]
            crop_file = crops[rng.randint(0, len(crops) - 1)]
            img = Image.open(crop_file).convert('RGB')
            img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        else:
            sign_file = self.sign_files[class_idx]
            sign_img = Image.open(sign_file).convert('RGBA')
            img = self._composite_sign(sign_img, seed)

        # Apply transforms
        if self.is_val:
            img = self.val_transform(img)
        else:
            img = self.post_transform(img)

        return img, class_idx


def train_classifier(
    signs_dir,
    backgrounds_dir=None,
    output_dir='classifier_model',
    img_size=96,
    samples_per_class=200,
    batch_size=64,
    epochs=30,
    lr=0.001,
    val_split=0.15,
    device='cuda',
    include_negative=True,
    negative_multiplier=3,
    extra_class_dirs=None,
):
    """
    Train MobileNetV3-Small classifier

    Args:
        signs_dir: Path to transparent PNG signs
        backgrounds_dir: Path to background images (optional)
        output_dir: Where to save model
        img_size: Input image size
        samples_per_class: Training samples per sign class
        batch_size: Batch size
        epochs: Training epochs
        lr: Learning rate
        val_split: Validation split ratio
        device: 'cuda' or 'cpu'
    """

    print("=" * 70)
    print("MobileNetV3-Small Traffic Sign Classifier Training")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create dataset
    print("\nLoading dataset...")
    dataset = SignPNGDataset(signs_dir, backgrounds_dir, samples_per_class, img_size,
                                include_negative, negative_multiplier, extra_class_dirs)

    # Split train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Set val transform for validation subset
    # We need a wrapper to apply val transforms
    class ValWrapper(Dataset):
        def __init__(self, subset, parent_dataset):
            self.subset = subset
            self.parent = parent_dataset

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            # Temporarily set val mode
            orig = self.parent.is_val
            self.parent.is_val = True
            item = self.subset[idx]
            self.parent.is_val = orig
            return item

    val_dataset_wrapped = ValWrapper(val_dataset, dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset_wrapped, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"Train: {train_size} samples")
    print(f"Val: {val_size} samples")
    print(f"Classes: {dataset.num_classes}")

    # Create model
    print("\nBuilding MobileNetV3-Small...")
    model = models.mobilenet_v3_small(pretrained=True)

    # Replace classifier head for our number of classes
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, dataset.num_classes)

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    print("=" * 70)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"  Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': dataset.class_names,
                'class_to_id': dataset.class_to_id,
                'num_classes': dataset.num_classes,
                'img_size': img_size,
                'val_acc': val_acc,
                'epoch': epoch + 1,
            }, output_path / 'best_classifier.pt')

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': dataset.class_names,
        'class_to_id': dataset.class_to_id,
        'num_classes': dataset.num_classes,
        'img_size': img_size,
        'val_acc': val_acc,
        'epoch': epochs,
    }, output_path / 'last_classifier.pt')

    # Save class names as JSON for easy loading
    with open(output_path / 'class_names.json', 'w') as f:
        json.dump(dataset.class_names, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_path / 'best_classifier.pt'}")
    print(f"Class names saved to: {output_path / 'class_names.json'}")

    return str(output_path / 'best_classifier.pt')


if __name__ == "__main__":
    print("\nTraffic Sign Classifier - MobileNetV3-Small")
    print("=" * 70)

    signs_dir = "data/transparent_png_signs"
    backgrounds_dir = "data/road_backgrounds"
    output_dir = "classifier_model"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    train_classifier(
        signs_dir=signs_dir,
        backgrounds_dir=backgrounds_dir,
        output_dir=output_dir,
        img_size=96,
        samples_per_class=200,
        batch_size=64,
        epochs=30,
        lr=0.001,
        device=device,
        include_negative=True,
        negative_multiplier=3,
        extra_class_dirs=[
            {'name': 'traffic_light', 'dir': 'data/traffic_light_crops'},
        ],
    )
