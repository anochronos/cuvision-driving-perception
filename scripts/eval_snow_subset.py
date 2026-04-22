"""Evaluate baseline vs fine-tuned model on BDD100K snow-labeled val images."""
import os
import sys
import json
import glob
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import TwinLiteNetPlus
from argparse import Namespace

BDD_ROOT = '../../capstone/bdd100k'
LABEL_DIR = os.path.join(BDD_ROOT, 'labels/100k/val')
IMG_DIR = os.path.join(BDD_ROOT, 'images/100k/val')
SEG_DIR = os.path.join(BDD_ROOT, 'images/bdd_seg_gt/val')
LANE_DIR = os.path.join(BDD_ROOT, 'images/bdd_lane_gt/val')


def find_snow_images():
    """Find all val images labeled as snowy weather."""
    snow_images = []
    for label_file in glob.glob(os.path.join(LABEL_DIR, '*.json')):
        with open(label_file) as f:
            data = json.load(f)
        weather = data.get('attributes', {}).get('weather', '')
        if weather == 'snowy':
            basename = os.path.basename(label_file).replace('.json', '')
            img_path = os.path.join(IMG_DIR, basename + '.jpg')
            seg_path = os.path.join(SEG_DIR, basename + '.png')
            lane_path = os.path.join(LANE_DIR, basename + '.png')
            if os.path.exists(img_path) and os.path.exists(seg_path) and os.path.exists(lane_path):
                snow_images.append((img_path, seg_path, lane_path))
    return snow_images


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im


def compute_metrics(pred, gt):
    """Compute IOU and pixel accuracy for binary masks."""
    pred_bin = (pred > 0.5).astype(np.float32)
    gt_bin = (gt > 0.5).astype(np.float32)

    intersection = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - intersection

    iou = intersection / (union + 1e-8)
    # mIOU: average of foreground IOU and background IOU
    bg_inter = ((1 - pred_bin) * (1 - gt_bin)).sum()
    bg_union = (1 - pred_bin).sum() + (1 - gt_bin).sum() - bg_inter
    bg_iou = bg_inter / (bg_union + 1e-8)
    miou = (iou + bg_iou) / 2.0

    correct = (pred_bin == gt_bin).sum()
    total = gt_bin.size
    acc = correct / total

    return miou, acc, iou


def evaluate_model(model, snow_images):
    """Run evaluation on snow subset."""
    W_, H_ = 640, 384
    da_mious, ll_accs, ll_ious = [], [], []

    model.eval()
    with torch.no_grad():
        for img_path, seg_path, lane_path in snow_images:
            image = cv2.imread(img_path)
            da_gt = cv2.imread(seg_path, 0)
            ll_gt = cv2.imread(lane_path, 0)

            if image is None or da_gt is None or ll_gt is None:
                continue

            # Preprocess image
            img = letterbox(image, (H_, W_))
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img_tensor = torch.from_numpy(img).unsqueeze(0).float().cuda() / 255.0

            # Inference
            da_pred, ll_pred = model(img_tensor)

            # Get predictions
            da_pred = torch.softmax(da_pred, dim=1)[:, 1, :, :].cpu().numpy()[0]
            ll_pred = torch.softmax(ll_pred, dim=1)[:, 1, :, :].cpu().numpy()[0]

            # Resize GT to match prediction size
            da_gt_resized = cv2.resize(da_gt, (W_, da_pred.shape[0]))
            ll_gt_resized = cv2.resize(ll_gt, (W_, ll_pred.shape[0]))
            _, da_gt_bin = cv2.threshold(da_gt_resized, 1, 1, cv2.THRESH_BINARY)
            _, ll_gt_bin = cv2.threshold(ll_gt_resized, 1, 1, cv2.THRESH_BINARY)

            da_miou, _, _ = compute_metrics(da_pred, da_gt_bin.astype(np.float32))
            ll_miou, ll_acc, ll_iou = compute_metrics(ll_pred, ll_gt_bin.astype(np.float32))

            da_mious.append(da_miou)
            ll_accs.append(ll_acc)
            ll_ious.append(ll_iou)

    return {
        'da_miou': np.mean(da_mious),
        'll_acc': np.mean(ll_accs),
        'll_iou': np.mean(ll_ious),
        'count': len(da_mious)
    }


def load_model(weights_path, config='large'):
    args = Namespace(config=config, onGPU=True)
    model = TwinLiteNetPlus(args).cuda()
    state_dict = torch.load(weights_path, map_location='cuda')
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    print("Finding snow-labeled val images...")
    snow_images = find_snow_images()
    print(f"Found {len(snow_images)} snow images in BDD100K val set\n")

    if len(snow_images) == 0:
        print("No snow images found. Check label directory.")
        return

    cudnn.benchmark = True

    # Evaluate baseline
    print("Evaluating BASELINE (large.pth) on snow subset...")
    baseline_model = load_model('trained-models/large.pth')
    baseline = evaluate_model(baseline_model, snow_images)
    print(f"  DA mIOU: {baseline['da_miou']:.4f}")
    print(f"  LL Acc:  {baseline['ll_acc']:.4f}")
    print(f"  LL IOU:  {baseline['ll_iou']:.4f}")
    del baseline_model
    torch.cuda.empty_cache()

    # Evaluate snow-only fine-tuned (best epoch = 29)
    best_ft = 'finetune_snow_only/model_29.pth'
    if not os.path.exists(best_ft):
        best_ft = 'finetune_snow_only/model_27.pth'
    print(f"\nEvaluating SNOW-ONLY FINE-TUNED ({best_ft}) on snow subset...")
    ft_model = load_model(best_ft)
    finetuned = evaluate_model(ft_model, snow_images)
    print(f"  DA mIOU: {finetuned['da_miou']:.4f}")
    print(f"  LL Acc:  {finetuned['ll_acc']:.4f}")
    print(f"  LL IOU:  {finetuned['ll_iou']:.4f}")
    del ft_model
    torch.cuda.empty_cache()

    # Deltas
    print(f"\n{'='*50}")
    print(f"SNOW SUBSET COMPARISON ({baseline['count']} images)")
    print(f"{'='*50}")
    d_da = finetuned['da_miou'] - baseline['da_miou']
    d_acc = finetuned['ll_acc'] - baseline['ll_acc']
    d_iou = finetuned['ll_iou'] - baseline['ll_iou']
    print(f"DA mIOU:  {baseline['da_miou']:.4f} → {finetuned['da_miou']:.4f}  ({'+' if d_da>=0 else ''}{d_da:.4f})")
    print(f"LL Acc:   {baseline['ll_acc']:.4f} → {finetuned['ll_acc']:.4f}  ({'+' if d_acc>=0 else ''}{d_acc:.4f})")
    print(f"LL IOU:   {baseline['ll_iou']:.4f} → {finetuned['ll_iou']:.4f}  ({'+' if d_iou>=0 else ''}{d_iou:.4f})")

    # ── Plot: Bar comparison ──
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['DA mIOU', 'LL Accuracy', 'LL IOU']
    before = [baseline['da_miou'], baseline['ll_acc'], baseline['ll_iou']]
    after = [finetuned['da_miou'], finetuned['ll_acc'], finetuned['ll_iou']]
    deltas = [a - b for a, b in zip(after, before)]

    x = np.arange(len(metrics))
    width = 0.3

    bars1 = ax.bar(x - width/2, before, width, label='Baseline (large.pth)', color='#e74c3c', alpha=0.85)
    bars2 = ax.bar(x + width/2, after, width, label='Snow-Only Fine-tuned', color='#2ecc71', alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar, d in zip(bars2, deltas):
        sign = '+' if d >= 0 else ''
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}\n({sign}{d:.3f})', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title(f'Before vs After: Snow-Only Fine-Tuning\n(BDD100K Val -SNOW-LABELED IMAGES ONLY, n={baseline["count"]})',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('docs/finetune_snow_only_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: docs/finetune_snow_only_comparison.png")

    # ── Plot: Training curves ──
    epochs = list(range(30))
    da_vals = [0.917, 0.918, 0.917, 0.918, 0.918, 0.917, 0.918, 0.918, 0.918, 0.919,
               0.918, 0.917, 0.918, 0.918, 0.918, 0.918, 0.918, 0.918, 0.919, 0.918,
               0.918, 0.919, 0.919, 0.918, 0.918, 0.918, 0.918, 0.919, 0.918, 0.919]
    ll_acc_vals = [0.778, 0.776, 0.779, 0.786, 0.783, 0.778, 0.783, 0.784, 0.784, 0.780,
                   0.778, 0.785, 0.782, 0.783, 0.781, 0.787, 0.787, 0.781, 0.783, 0.785,
                   0.784, 0.784, 0.783, 0.784, 0.784, 0.786, 0.785, 0.785, 0.785, 0.785]
    ll_iou_vals = [0.316, 0.318, 0.316, 0.318, 0.318, 0.317, 0.317, 0.316, 0.317, 0.316,
                   0.317, 0.318, 0.318, 0.317, 0.317, 0.318, 0.318, 0.317, 0.316, 0.317,
                   0.317, 0.318, 0.318, 0.318, 0.318, 0.318, 0.318, 0.318, 0.318, 0.319]

    fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle('Snow-Only Fine-Tuning: Training Curves (val on 769 snow images)',
                  fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(epochs, da_vals, 'b-o', markersize=3, linewidth=1.5)
    ax.set_title('DA mIOU')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mIOU')
    ax.set_ylim(0.915, 0.921)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, ll_acc_vals, 'g-o', markersize=3, linewidth=1.5)
    ax.set_title('LL Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.773, 0.790)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, ll_iou_vals, 'm-o', markersize=3, linewidth=1.5)
    ax.set_title('LL IOU')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IOU')
    ax.set_ylim(0.314, 0.321)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('docs/finetune_snow_only_curves.png', dpi=150, bbox_inches='tight')
    print(f"Saved: docs/finetune_snow_only_curves.png")


if __name__ == '__main__':
    main()
