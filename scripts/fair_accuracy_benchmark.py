"""Fair accuracy benchmark: TLNP uses its own val() from utils.py,
YOLOPX uses its own validate() pipeline. Both on full BDD100K val."""
import sys, os, json, subprocess, yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np

TLNP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLOPX_DIR = '/home/akshun-wsl/capstone-yolopx/YOLOPX'

sys.path.insert(0, TLNP_DIR)
os.chdir(TLNP_DIR)

def eval_tlnp():
    """Use TLNP's own val() with its Dataset and SegmentationMetric."""
    from argparse import Namespace
    from model.model import TwinLiteNetPlus
    from utils import val
    import BDD100K

    with open('hyperparameters/twinlitev2_hyper.yaml') as f:
        hyp = yaml.safe_load(f)

    args = Namespace(config='large', onGPU=True, verbose=False)
    model = TwinLiteNetPlus(args).cuda().eval()
    model.load_state_dict(torch.load('trained-models/large.pth', map_location='cuda'))

    valLoader = torch.utils.data.DataLoader(
        BDD100K.Dataset(hyp, valid=True),
        batch_size=16, shuffle=False, num_workers=12, pin_memory=True)

    print(f"  TLNP val loader: {len(valLoader)} batches, {len(valLoader.dataset)} images")

    da_result, ll_result = val(valLoader, model, args=args)

    return {
        "da_acc": round(float(da_result[0]), 4),
        "da_iou": round(float(da_result[1]), 4),
        "da_miou": round(float(da_result[2]), 4),
        "ll_acc": round(float(ll_result[0]), 4),
        "ll_iou": round(float(ll_result[1]), 4),
        "ll_miou": round(float(ll_result[2]), 4),
    }


def eval_yolopx(conda_env):
    """Run YOLOPX's own validation using its test.py pipeline."""
    code = """
import sys, os, json, torch, numpy as np
sys.path.insert(0, '{ydir}')
os.chdir('{ydir}')

import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from lib.config import cfg
from lib.models import get_net
from lib.utils import DataLoaderX
from lib.core.function import validate, AverageMeter
from lib.core.general import fitness
import lib.dataset as dataset
from lib.utils.utils import create_logger, select_device
from lib.core.loss import get_loss

cudnn.benchmark = True

# Use YOLOPX's own data loading
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Build model
device = torch.device('cuda:0')
model = get_net(cfg)
model = model.to(device)

# Load weights
checkpoint = torch.load('weights/epoch-195.pth', map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Build val dataset using YOLOPX's own dataset class
valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
    cfg=cfg,
    is_train=False,
    inputsize=cfg.MODEL.IMAGE_SIZE,
    transform=transforms.Compose([transforms.ToTensor(), normalize])
)

valid_loader = DataLoaderX(
    valid_dataset,
    batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
    shuffle=False,
    num_workers=cfg.WORKERS,
    pin_memory=True,
    collate_fn=dataset.AutoDriveDataset.collate_fn
)

print(f'YOLOPX val loader: {{len(valid_loader)}} batches, {{len(valid_dataset)}} images', flush=True)

# Use YOLOPX's own SegmentationMetric
from lib.core.function import SegmentationMetric
from tqdm import tqdm

da_metric = SegmentationMetric(cfg.num_seg_class)
ll_metric = SegmentationMetric(2)

da_acc_seg = AverageMeter()
da_IoU_seg = AverageMeter()
da_mIoU_seg = AverageMeter()
ll_acc_seg = AverageMeter()
ll_IoU_seg = AverageMeter()
ll_mIoU_seg = AverageMeter()

with torch.no_grad():
    for batch_i, (img, target, paths, shapes) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        img = img.to(device, non_blocking=True)
        assign_target = []
        for tgt in target:
            assign_target.append(tgt.to(device))
        target = assign_target
        nb, _, height, width = img.shape

        pad_w, pad_h = shapes[0][1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)

        det_out, da_seg_out, ll_seg_out = model(img)

        # DA eval
        _, da_predict = torch.max(da_seg_out, 1)
        _, da_gt = torch.max(target[1], 1)
        da_predict = da_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
        da_gt = da_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

        da_metric.reset()
        da_metric.addBatch(da_predict.cpu(), da_gt.cpu())
        da_acc = da_metric.pixelAccuracy()
        da_IoU_arr = da_metric.IntersectionOverUnion()
        da_IoU = np.mean(da_IoU_arr[1:])
        da_mIoU = da_metric.meanIntersectionOverUnion()
        da_acc_seg.update(da_acc, img.size(0))
        da_IoU_seg.update(da_IoU, img.size(0))
        da_mIoU_seg.update(da_mIoU, img.size(0))

        # LL eval
        _, ll_predict = torch.max(ll_seg_out, 1)
        _, ll_gt = torch.max(target[2], 1)
        ll_predict = ll_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
        ll_gt = ll_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

        ll_metric.reset()
        ll_metric.addBatch(ll_predict.cpu(), ll_gt.cpu())
        ll_acc = ll_metric.lineAccuracy()
        ll_IoU = ll_metric.IntersectionOverUnion()
        ll_mIoU = ll_metric.meanIntersectionOverUnion()
        ll_acc_seg.update(ll_acc, img.size(0))
        ll_IoU_seg.update(ll_IoU, img.size(0))
        ll_mIoU_seg.update(ll_mIoU, img.size(0))

r = {{
    "da_acc": round(float(da_acc_seg.avg), 4),
    "da_iou": round(float(da_IoU_seg.avg), 4),
    "da_miou": round(float(da_mIoU_seg.avg), 4),
    "ll_acc": round(float(ll_acc_seg.avg), 4),
    "ll_iou": round(float(ll_IoU_seg.avg), 4),
    "ll_miou": round(float(ll_mIoU_seg.avg), 4),
}}
print('YOLOPX_ACC:' + json.dumps(r))
""".format(ydir=YOLOPX_DIR)

    tmp = '/tmp/bench_yolopx_faireval.py'
    with open(tmp, 'w') as f:
        f.write(code)

    cmd = f'conda run -n {conda_env} --no-capture-output python {tmp}'
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1200)

    if r.returncode != 0:
        print(f"  YOLOPX error: {r.stderr[:800]}")
        return None

    for line in r.stdout.split('\n'):
        if 'YOLOPX_ACC:' in line:
            return json.loads(line.split('YOLOPX_ACC:')[1])

    print(f"  No result. stdout tail: {r.stdout[-500:]}")
    return None


if __name__ == '__main__':
    conda_env = sys.argv[1] if len(sys.argv) > 1 else 'yolopx'
    cudnn.benchmark = True

    print("=" * 60)
    print("FAIR ACCURACY BENCHMARK: TLNP-Large vs YOLOPX")
    print("Each model uses its OWN validation pipeline")
    print("Full BDD100K val set (10,000 images)")
    print("=" * 60)

    print("\n[1/2] Evaluating TwinLiteNetPlus-Large...")
    tlnp = eval_tlnp()
    print(f"\n  TLNP Results:")
    print(f"    DA: Acc={tlnp['da_acc']:.4f}  IOU={tlnp['da_iou']:.4f}  mIOU={tlnp['da_miou']:.4f}")
    print(f"    LL: Acc={tlnp['ll_acc']:.4f}  IOU={tlnp['ll_iou']:.4f}  mIOU={tlnp['ll_miou']:.4f}")

    torch.cuda.empty_cache()

    print("\n[2/2] Evaluating YOLOPX...")
    yolopx = eval_yolopx(conda_env)
    if yolopx:
        print(f"\n  YOLOPX Results:")
        print(f"    DA: Acc={yolopx['da_acc']:.4f}  IOU={yolopx['da_iou']:.4f}  mIOU={yolopx['da_miou']:.4f}")
        print(f"    LL: Acc={yolopx['ll_acc']:.4f}  IOU={yolopx['ll_iou']:.4f}  mIOU={yolopx['ll_miou']:.4f}")

    # Final comparison
    print(f"\n{'='*70}")
    print(f"  FAIR ACCURACY COMPARISON (each model's own eval pipeline)")
    print(f"{'='*70}")
    print(f"  {'Metric':<20} {'TLNP-Large':>15} {'YOLOPX':>15}")
    print(f"  {'-'*50}")
    print(f"  {'DA Accuracy':<20} {tlnp['da_acc']:>15.4f} {yolopx['da_acc'] if yolopx else 'N/A':>15}")
    print(f"  {'DA IOU':<20} {tlnp['da_iou']:>15.4f} {yolopx['da_iou'] if yolopx else 'N/A':>15}")
    print(f"  {'DA mIOU':<20} {tlnp['da_miou']:>15.4f} {yolopx['da_miou'] if yolopx else 'N/A':>15}")
    print(f"  {'LL Accuracy':<20} {tlnp['ll_acc']:>15.4f} {yolopx['ll_acc'] if yolopx else 'N/A':>15}")
    print(f"  {'LL IOU':<20} {tlnp['ll_iou']:>15.4f} {yolopx['ll_iou'] if yolopx else 'N/A':>15}")
    print(f"  {'LL mIOU':<20} {tlnp['ll_miou']:>15.4f} {yolopx['ll_miou'] if yolopx else 'N/A':>15}")
    print(f"  {'-'*50}")

    # Save
    results = {"tlnp": tlnp, "yolopx": yolopx}
    with open('docs/fair_accuracy_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to: docs/fair_accuracy_benchmark.json")
