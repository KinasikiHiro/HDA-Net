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
# Dataset loader (3D-based)
# ----------------------
class TestH5Dataset:
    def __init__(self, folder):
        self.files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".h5")])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        with h5py.File(self.files[idx], "r") as f:
            img = f["image"][()]  # (H, W, D)
            label = f["label"][()]  # (H, W, D)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = np.expand_dims(img, axis=0)  # -> (1, H, W, D)
        label = np.expand_dims(label, axis=0)
        return img, label, os.path.basename(self.files[idx])

# ----------------------
# Evaluation (single model)
# ----------------------
def evaluate_model(model, dataset, device, num_classes=3):
    model.eval()
    results = []
    with torch.no_grad():
        for img, label, name in tqdm(dataset, desc="Evaluating"):
            img_t = torch.from_numpy(img).unsqueeze(0).float().to(device)  # (B=1, C=1, H, W, D)
            out = model(img_t)
            pred = torch.argmax(out, dim=1).cpu().numpy()[0]
            gt = label[0]

            metrics = {}
            for cls in range(1, num_classes):
                p = (pred == cls).astype(np.float32)
                t = (gt == cls).astype(np.float32)
                metrics[f"dice_cls{cls}"] = dice(p, t)
                metrics[f"iou_cls{cls}"] = iou(p, t)
                metrics[f"precision_cls{cls}"] = precision(p, t)
                metrics[f"recall_cls{cls}"] = recall(p, t)
                metrics[f"vol_diff_cls{cls}"] = volume_diff(p, t)
            results.append({"case": name, **metrics})
    return results

# ----------------------
# Visualization
# ----------------------
def plot_boxplot(results_a, results_b, out_dir, metrics=["dice_cls1","iou_cls1","precision_cls1","recall_cls1"]):
    plt.figure(figsize=(10,5))
    data_a = [[r[m] for r in results_a] for m in metrics]
    data_b = [[r[m] for r in results_b] for m in metrics]
    plt.boxplot(data_a, positions=np.arange(len(metrics))*2.0, widths=0.6, labels=metrics)
    plt.boxplot(data_b, positions=np.arange(len(metrics))*2.0+0.8, widths=0.6)
    plt.title("Baseline (blue) vs Joint (orange)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_boxplot.png"), dpi=300)
    plt.close()

def plot_radar(avg_a, avg_b, out_dir):
    labels = list(avg_a.keys())
    n = len(labels)
    angles = np.linspace(0, 2*pi, n, endpoint=False).tolist()
    avg_a_vals = list(avg_a.values())
    avg_b_vals = list(avg_b.values())
    avg_a_vals += avg_a_vals[:1]
    avg_b_vals += avg_b_vals[:1]
    angles += angles[:1]
    plt.figure(figsize=(6,6))
    plt.polar(angles, avg_a_vals, label="Baseline", color="blue")
    plt.polar(angles, avg_b_vals, label="Joint", color="orange")
    plt.fill(angles, avg_a_vals, alpha=0.25, color="blue")
    plt.fill(angles, avg_b_vals, alpha=0.25, color="orange")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "comparison_radar.png"), dpi=300)
    plt.close()

# ----------------------
# Helper: Save & Load metrics
# ----------------------
def save_metrics_to_csv(results, csv_path, model_name):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        keys = ["case"] + [k for k in results[0].keys() if k != "case"]
        writer.writerow(["model"] + keys)
        for r in results:
            writer.writerow([model_name] + [r[k] for k in keys])

def load_csv_as_dict(csv_path):
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)
    results = []
    for r in rows:
        case_dict = {"case": r[1]}
        for i, k in enumerate(rows[0][2:], start=2):
            case_dict[k] = float(r[i])
        results.append(case_dict)
    return results

# ----------------------
# Main
# ----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Using device: {device}")

    test_dir = os.path.join("..", "preprocessed", "test")
    results_dir = os.path.join("..", "results", "joint_evaluate")
    os.makedirs(results_dir, exist_ok=True)
    dataset = TestH5Dataset(test_dir)

    # ---- 1️⃣ Baseline Evaluation ----
    print("[Eval] Stage 1: Evaluating baseline model...")
    base_model = HybridHDenseUNet(encoder_out_ch=64, n_classes=3).to(device)
    base_model.load_state_dict(torch.load(os.path.join("..","models","best_h_denseunet_model.pt"), map_location=device, weights_only=True))
    base_results = evaluate_model(base_model, dataset, device)
    save_metrics_to_csv(base_results, os.path.join(results_dir, "baseline_metrics.csv"), "baseline")
    del base_model
    torch.cuda.empty_cache()

    # ---- 2️⃣ Joint Evaluation ----
    print("[Eval] Stage 2: Evaluating joint model...")
    joint_model = HybridHDenseUNet(encoder_out_ch=64, n_classes=3).to(device)
    joint_ckpt = torch.load(os.path.join("..","models","joint_seg_best.pt"), map_location=device, weights_only=True)
    joint_model.load_state_dict(joint_ckpt["model"] if "model" in joint_ckpt else joint_ckpt)
    joint_results = evaluate_model(joint_model, dataset, device)
    save_metrics_to_csv(joint_results, os.path.join(results_dir, "joint_metrics.csv"), "joint")
    del joint_model
    torch.cuda.empty_cache()

    # ---- 3️⃣ Summary & Visualization ----
    def avg_std(results):
        out = {}
        for k in results[0].keys():
            if k == "case": continue
            vals = [r[k] for r in results]
            out[k] = np.mean(vals)
        return out

    avg_base = avg_std(base_results)
    avg_joint = avg_std(joint_results)

    summary_txt = os.path.join(results_dir, "summary.txt")
    with open(summary_txt, "w") as f:
        f.write("==== Evaluation Summary ====\n")
        f.write(f"Source-only Dice: {avg_base['dice_cls1']:.4f}\n")
        f.write(f"Joint-adapted Dice: {avg_joint['dice_cls1']:.4f}\n")
        f.write(f"DGRR: {avg_joint['dice_cls1']-avg_base['dice_cls1']:.4f}\n\n")
        for k in avg_base.keys():
            base = avg_base[k]; joint = avg_joint[k]
            f.write(f"{k:<12} src={base:.4f}, joint={joint:.4f}\n")
        f.write("============================\n")

    plot_boxplot(base_results, joint_results, results_dir)
    plot_radar(avg_base, avg_joint, results_dir)

    print("[Eval] ✅ All evaluation finished and visualizations saved!")

if __name__ == "__main__":
    main()
