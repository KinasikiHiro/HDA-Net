import os
import sys
import csv
import torch
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import pi

# ----------------------
# Path setup
# ----------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
UTILS_DIR = os.path.join(ROOT_DIR, "utils")
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "experiments")
if EXPERIMENTS_DIR not in sys.path:
    sys.path.append(EXPERIMENTS_DIR)

from train_h_denseunet import HybridHDenseUNet

# ----------------------
# Metric functions
# ----------------------
EPS = 1e-6
def dice(pred, target): inter = (pred * target).sum(); return ((2 * inter + EPS) / (pred.sum() + target.sum() + EPS)).item()
def iou(pred, target): inter = (pred * target).sum(); union = pred.sum() + target.sum() - inter; return ((inter + EPS) / (union + EPS)).item()
def precision(pred, target): tp = (pred * target).sum(); fp = (pred * (1 - target)).sum(); return ((tp + EPS) / (tp + fp + EPS)).item()
def recall(pred, target): tp = (pred * target).sum(); fn = ((1 - pred) * target).sum(); return ((tp + EPS) / (tp + fn + EPS)).item()
def volume_diff(pred, target): return abs(pred.sum().item() - target.sum().item()) / (target.sum().item() + EPS)

# ----------------------
# Dataset loader (case-level)
# ----------------------
class TestH5Dataset:
    def __init__(self, folder):
        self.files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".h5")])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        file = self.files[idx]
        with h5py.File(file, "r") as f:
            img = f["image"][()]  # (H, W, D)
            label = f["label"][()] if "label" in f else np.zeros_like(img, dtype=np.uint8)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        # ✅ 关键修改：transpose label -> (D,H,W) 与模型输出对齐
        label = label.transpose(2,0,1)
        case_id = os.path.splitext(os.path.basename(file))[0]
        # 返回 (B=1,C=1,D,H,W)
        return torch.from_numpy(img[None,None]).float(), torch.from_numpy(label).long(), case_id

# ----------------------
# Patch inference
# ----------------------
def sliding_window_inference(model, volume, patch_size, overlap=0.25):
    model.eval()
    B, C, D, H, W = volume.shape
    stride_d = int(patch_size[0] * (1 - overlap))
    stride_h = int(patch_size[1] * (1 - overlap))
    stride_w = int(patch_size[2] * (1 - overlap))

    output = torch.zeros((B, model.n_classes, D, H, W), device=volume.device)
    counts = torch.zeros((B, 1, D, H, W), device=volume.device)

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for z in range(0, D - patch_size[0] + 1, stride_d):
            for y in range(0, H - patch_size[1] + 1, stride_h):
                for x in range(0, W - patch_size[2] + 1, stride_w):
                    patch = volume[:, :, z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]]
                    out_patch = model(patch)
                    output[:, :, z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] += out_patch
                    counts[:, :, z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] += 1

        output /= counts
    return output

# ----------------------
# Evaluation
# ----------------------
def evaluate_model(model, dataset, device, patch_size=(16,128,128)):
    model.eval()
    results = []

    with torch.no_grad():
        for img_t, label_t, case_id in tqdm(dataset, desc="Evaluating"):
            torch.cuda.empty_cache()
            img_t = img_t.to(device, non_blocking=True)  # (1,1,D,H,W)
            label_t = label_t.numpy()  # (D,H,W)

            try:
                with torch.amp.autocast('cuda'):
                    if patch_size is not None and img_t.ndim == 5:
                        out = sliding_window_inference(model, img_t, patch_size)
                    else:
                        out = model(img_t)
                # ✅ 关键修改：确保 pred shape == label shape
                pred = torch.argmax(out, dim=1).cpu().numpy()[0]  # (D,H,W)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"[WARN] OOM on {case_id}, retrying with smaller patch size...")
                    torch.cuda.empty_cache()
                    with torch.amp.autocast('cuda'):
                        out = sliding_window_inference(model, img_t, (8,96,96))
                    pred = torch.argmax(out, dim=1).cpu().numpy()[0]
                else:
                    raise e

            metrics = {}
            for cls in range(1, 3):  # liver/tumor
                p = (pred == cls).astype(np.float32)
                t = (label_t == cls).astype(np.float32)
                metrics[f"dice_cls{cls}"] = dice(p, t)
                metrics[f"iou_cls{cls}"] = iou(p, t)
                metrics[f"precision_cls{cls}"] = precision(p, t)
                metrics[f"recall_cls{cls}"] = recall(p, t)
                metrics[f"vol_diff_cls{cls}"] = volume_diff(p, t)

            results.append({"case": case_id, **metrics})

            del img_t, out
            torch.cuda.empty_cache()

    return results

# ----------------------
# Main: 和你原来的保持一致
# ----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Using device: {device}")

    test_dir = os.path.join("..","preprocessed","test")
    results_dir = os.path.join("..","results","joint_evaluate")
    os.makedirs(results_dir, exist_ok=True)
    dataset = TestH5Dataset(test_dir)

    # ---------------- Stage 1: baseline ----------------
    print("[Eval] Stage 1: Evaluating baseline model...")
    base_model = HybridHDenseUNet(encoder_out_ch=64, n_classes=3).to(device)
    base_model.n_classes = 3
    base_model.load_state_dict(
        torch.load(os.path.join("..","models","best_h_denseunet_model.pt"), map_location=device, weights_only=True)
    )
    base_results = evaluate_model(base_model, dataset, device, patch_size=(16,128,128))
    del base_model; torch.cuda.empty_cache()

    # ---------------- Stage 2: joint ----------------
    print("[Eval] Stage 2: Evaluating joint model...")
    joint_model = HybridHDenseUNet(encoder_out_ch=64, n_classes=3).to(device)
    joint_model.n_classes = 3
    joint_ckpt = torch.load(os.path.join("..","models","joint_seg_best.pt"), map_location=device, weights_only=True)
    joint_model.load_state_dict(joint_ckpt["model"] if "model" in joint_ckpt else joint_ckpt)
    joint_results = evaluate_model(joint_model, dataset, device, patch_size=(16,128,128))
    del joint_model; torch.cuda.empty_cache()

    # ---------------- Stage 3: save metrics ----------------
    csv_path = os.path.join(results_dir,"metrics.csv")
    with open(csv_path,"w",newline="") as f:
        writer = csv.writer(f)
        keys = ["case"] + [k for k in base_results[0].keys() if k!="case"]
        writer.writerow(["model"] + keys)
        for br,jr in zip(base_results, joint_results):
            writer.writerow(["baseline"] + [br[k] for k in keys])
            writer.writerow(["joint"] + [jr[k] for k in keys])
    print(f"[Eval] Metrics saved → {csv_path}")

    # ---------------- Stage 4: summary ----------------
    def avg_std(results):
        out = {}
        for k in results[0].keys():
            if k=="case": continue
            vals = [r[k] for r in results]
            out[k] = np.mean(vals)
        return out
    avg_base = avg_std(base_results)
    avg_joint = avg_std(joint_results)

    summary_txt = os.path.join(results_dir,"summary.txt")
    with open(summary_txt,"w") as f:
        f.write("==== Evaluation Summary ====\n")
        f.write(f"Source-only Dice: {avg_base['dice_cls1']:.4f}\n")
        f.write(f"Joint-adapted Dice: {avg_joint['dice_cls1']:.4f}\n")
        f.write(f"DGRR: {avg_joint['dice_cls1']-avg_base['dice_cls1']:.4f}\n\n")
        for k in avg_base.keys():
            base = avg_base[k]; joint = avg_joint[k]
            f.write(f"{k:<12} src={base:.4f}, joint={joint:.4f}\n")
        f.write("============================\n")
    print(f"[Eval] Summary saved → {summary_txt}")

if __name__=="__main__":
    main()
