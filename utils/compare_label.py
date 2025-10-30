# compare_labels_safe.py
import os
import csv
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

EPS = 1e-6

# ----------------------
# Metric functions
# ----------------------
def dice(pred, target):
    inter = (pred * target).sum()
    return ((2 * inter + EPS) / (pred.sum() + target.sum() + EPS)).item()

def iou(pred, target):
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return ((inter + EPS) / (union + EPS)).item()

def precision(pred, target):
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return ((tp + EPS) / (tp + fp + EPS)).item()

def recall(pred, target):
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return ((tp + EPS) / (tp + fn + EPS)).item()

def volume_diff(pred, target):
    return abs(pred.sum().item() - target.sum().item()) / (target.sum().item() + EPS)

# ----------------------
# H5 loader
# ----------------------
def load_h5(path, key="label"):
    with h5py.File(path, "r") as f:
        return f[key][()]

# ----------------------
# Resize function (nearest for labels)
# ----------------------
def resize_label(label, target_shape):
    """Resize 3D label to target shape (H,W,D) using nearest interpolation"""
    if label.shape == target_shape:
        return label
    # shape = (H,W,D) -> add batch+channel for F.interpolate
    label_tensor = torch.from_numpy(label[None,None].astype(np.float32))
    target_tensor = F.interpolate(label_tensor, size=target_shape, mode='nearest')
    return target_tensor[0,0].numpy().astype(np.uint8)

# ----------------------
# Main comparison
# ----------------------
def main():
    # Paths
    PREP_TEST_DIR = os.path.join("..", "preprocessed", "test")
    GEN_LABEL_DIR = os.path.join("..", "preprocessed", "pseudo_labels")
    OUT_DIR = os.path.join("..", "results", "label_comparison")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Find files (一一对应)
    test_files = sorted([f for f in os.listdir(PREP_TEST_DIR) if f.endswith(".h5")])
    gen_files = sorted([f for f in os.listdir(GEN_LABEL_DIR) if f.endswith(".h5")])

    if len(test_files) != len(gen_files):
        print("[WARN] Test and generated label counts differ!")

    results = []
    for t_file, g_file in tqdm(zip(test_files, gen_files), total=len(gen_files), desc="Comparing labels"):
        true_path = os.path.join(PREP_TEST_DIR, t_file)
        gen_path = os.path.join(GEN_LABEL_DIR, g_file)

        # Load
        true_label = load_h5(true_path, "label")
        gen_label = load_h5(gen_path, "label")

        # Align shape
        gen_label = resize_label(gen_label, true_label.shape)

        case_metrics = {"case": t_file}
        for cls, name in zip([1,2], ["Liver","Tumor"]):
            p = (gen_label == cls).astype(np.float32)
            t = (true_label == cls).astype(np.float32)
            case_metrics[f"dice_{name}"] = dice(p, t)
            case_metrics[f"iou_{name}"] = iou(p, t)
            case_metrics[f"precision_{name}"] = precision(p, t)
            case_metrics[f"recall_{name}"] = recall(p, t)
            case_metrics[f"vol_diff_{name}"] = volume_diff(p, t)
        results.append(case_metrics)

    # ----------------------
    # Save CSV
    # ----------------------
    csv_path = os.path.join(OUT_DIR, "label_metrics.csv")
    keys = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"[INFO] CSV saved → {csv_path}")

    # ----------------------
    # Save summary
    # ----------------------
    summary_txt = os.path.join(OUT_DIR, "summary.txt")
    avg_metrics = {}
    for key in results[0].keys():
        if key=="case": continue
        avg_metrics[key] = np.mean([r[key] for r in results])

    with open(summary_txt, "w") as f:
        f.write("==== Label Comparison Summary ====\n")
        for k,v in avg_metrics.items():
            f.write(f"{k:<15}: {v:.4f}\n")
        f.write("===============================\n")
    print(f"[INFO] Summary saved → {summary_txt}")

if __name__=="__main__":
    main()
