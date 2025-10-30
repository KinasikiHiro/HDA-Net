import os
import sys
import csv
import time
import traceback
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
            label = f["label"][()]  # (H, W, D)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        case_id = os.path.splitext(os.path.basename(file))[0]
        # return tensor image (1,1,H,W,D) and numpy label (H,W,D) for metrics
        return torch.from_numpy(img[None, None]).float(), torch.from_numpy(label).long(), case_id

# ----------------------
# Patch inference (sliding window) with AMP
# ----------------------
def sliding_window_inference(model, volume, patch_size, overlap=0.25, device='cuda'):
    """
    volume: tensor (B, C, D, H, W)
    patch_size: tuple (pD, pH, pW)
    returns: tensor (B, n_classes, D, H, W)
    """
    model.eval()
    B, C, D, H, W = volume.shape
    pD, pH, pW = patch_size
    stride_d = max(1, int(pD * (1 - overlap)))
    stride_h = max(1, int(pH * (1 - overlap)))
    stride_w = max(1, int(pW * (1 - overlap)))

    n_classes = getattr(model, "n_classes", None)
    if n_classes is None:
        # fallback: try to infer from model final conv if possible (not guaranteed)
        # user sets model.n_classes before calling evaluate (we do too in main)
        raise AttributeError("Model must have attribute 'n_classes' set before calling sliding_window_inference")

    output = torch.zeros((B, n_classes, D, H, W), device=volume.device)
    counts = torch.zeros((B, 1, D, H, W), device=volume.device)

    with torch.no_grad(), torch.cuda.amp.autocast('cuda'):
        z_ranges = list(range(0, max(1, D - pD + 1), stride_d))
        if z_ranges[-1] != D - pD:
            z_ranges.append(max(0, D - pD))
        y_ranges = list(range(0, max(1, H - pH + 1), stride_h))
        if y_ranges[-1] != H - pH:
            y_ranges.append(max(0, H - pH))
        x_ranges = list(range(0, max(1, W - pW + 1), stride_w))
        if x_ranges[-1] != W - pW:
            x_ranges.append(max(0, W - pW))

        for z in z_ranges:
            for y in y_ranges:
                for x in x_ranges:
                    patch = volume[:, :, z:z + pD, y:y + pH, x:x + pW]
                    out_patch = model(patch)  # expect (B, n_classes, pD, pH, pW)
                    # If returned spatial dims mismatch (rare), try to interpolate to patch size
                    if out_patch.shape[2:] != (pD, pH, pW):
                        out_patch = torch.nn.functional.interpolate(out_patch, size=(pD, pH, pW), mode='trilinear', align_corners=False)
                    output[:, :, z:z + pD, y:y + pH, x:x + pW] += out_patch
                    counts[:, :, z:z + pD, y:y + pH, x:x + pW] += 1.0

        # Avoid division by zero
        counts[counts == 0] = 1.0
        output = output / counts

    return output

# ----------------------
# Evaluate (per-case safe, OOM handling + logging)
# ----------------------
def evaluate_model(model, dataset, device, patch_size=(16, 128, 128), min_patch=(4,64,64), max_retries=3):
    model.eval()
    results = []
    failed_cases = []

    log_path = os.path.join(ROOT_DIR, "results", "joint_evaluate", "eval_error_log.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # start log
    with open(log_path, "w") as lf:
        lf.write(f"[Eval Log] Started at {time.asctime()}\n")
        lf.write(f"Patch size start: {patch_size}, min_patch: {min_patch}\n\n")

    for img_t, label_t, case_id in tqdm(dataset, desc="Evaluating"):
        case_start = time.time()
        try:
            # ensure GPU memory clean before case
            torch.cuda.empty_cache()

            img_t = img_t.to(device, non_blocking=True)  # (1,1,H,W,D) or (1,1,D,H,W) depending on layout - matches your training
            # ensure model.n_classes exists
            if not hasattr(model, "n_classes"):
                raise AttributeError("model.n_classes is not set (please set e.g. model.n_classes = 3)")

            # Try inference with progressive fallback when OOM
            current_patch = tuple(patch_size)
            success = False
            attempt = 0
            last_exc = None

            while attempt < max_retries and not success:
                try:
                    with torch.no_grad(), torch.cuda.amp.autocast('cuda'):
                        if img_t.ndim == 5:
                            out = sliding_window_inference(model, img_t, current_patch)
                        else:
                            out = model(img_t)  # fallback: whole-volume inference
                    success = True
                except RuntimeError as e:
                    last_exc = e
                    if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                        # OOM: reduce patch size and retry
                        attempt += 1
                        with open(log_path, "a") as lf:
                            lf.write(f"[{time.asctime()}] OOM on case {case_id} attempt {attempt} with patch {current_patch}. Exception: {repr(e)}\n")
                        torch.cuda.empty_cache()
                        # reduce patch by half each dimension but not below min_patch
                        new_p = (
                            max(min_patch[0], current_patch[0] // 2),
                            max(min_patch[1], current_patch[1] // 2),
                            max(min_patch[2], current_patch[2] // 2)
                        )
                        # if no change anymore -> break
                        if new_p == current_patch:
                            break
                        current_patch = new_p
                        time.sleep(0.5)
                        continue
                    else:
                        # non-OOM runtime error: re-raise after logging
                        with open(log_path, "a") as lf:
                            lf.write(f"[{time.asctime()}] Runtime error on case {case_id}: {traceback.format_exc()}\n")
                        raise

            if not success:
                # final failure for this case
                failed_cases.append(case_id)
                with open(log_path, "a") as lf:
                    lf.write(f"[{time.asctime()}] SKIPPED case {case_id} after {attempt} attempts. Last exception:\n")
                    lf.write(traceback.format_exc() + "\n")
                # free mem and continue
                try:
                    del img_t, out
                except:
                    pass
                torch.cuda.empty_cache()
                continue

            # produce prediction (B=1)
            pred = torch.argmax(out, dim=1).cpu().numpy()[0]  # (D,H,W) or (H,W,D) depending layout
            # ensure label numpy
            gt = label_t.numpy()

            # compute metrics for classes 1..(n_classes-1)
            metrics = {}
            n_classes = model.n_classes
            for cls in range(1, n_classes):
                p = (pred == cls).astype(np.float32)
                t = (gt == cls).astype(np.float32)
                metrics[f"dice_cls{cls}"] = dice(p, t)
                metrics[f"iou_cls{cls}"] = iou(p, t)
                metrics[f"precision_cls{cls}"] = precision(p, t)
                metrics[f"recall_cls{cls}"] = recall(p, t)
                metrics[f"vol_diff_cls{cls}"] = volume_diff(p, t)

            results.append({"case": case_id, **metrics})

            # cleanup per-case
            del img_t, out, pred, gt
            torch.cuda.empty_cache()
            case_time = time.time() - case_start
            with open(log_path, "a") as lf:
                lf.write(f"[{time.asctime()}] SUCCESS case {case_id} (patch_used={current_patch}) time={case_time:.1f}s\n")

        except Exception as e:
            # any unexpected exception: log and skip
            failed_cases.append(case_id)
            with open(log_path, "a") as lf:
                lf.write(f"[{time.asctime()}] EXCEPTION while processing case {case_id}:\n")
                lf.write(traceback.format_exc() + "\n\n")
            # attempt to free memory and continue
            try:
                del img_t
            except:
                pass
            torch.cuda.empty_cache()
            continue

    # final log summary
    with open(log_path, "a") as lf:
        lf.write(f"\n[Eval Log] Finished at {time.asctime()}\n")
        lf.write(f"Total cases processed: {len(results)}; failed/skipped: {len(failed_cases)}\n")
        if failed_cases:
            lf.write("Failed cases:\n")
            for c in failed_cases:
                lf.write(f" - {c}\n")

    print(f"[Eval] Done. Processed {len(results)} cases. Skipped {len(failed_cases)} cases. See {log_path} for details.")
    return results

# ----------------------
# Visualization (unchanged)
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
# Main
# ----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Using device: {device}")

    test_dir = os.path.join("..", "preprocessed", "test")
    results_dir = os.path.join("..", "results", "joint_evaluate")
    os.makedirs(results_dir, exist_ok=True)
    dataset = TestH5Dataset(test_dir)

    # ---------------- Stage 1: baseline ----------------
    print("[Eval] Stage 1: Evaluating baseline model...")
    base_model = HybridHDenseUNet(encoder_out_ch=64, n_classes=3).to(device)
    base_model.n_classes = 3  # ensure attribute exists for sliding window
    # try to load either state_dict or wrapper checkpoint
    base_ckpt_path = os.path.join("..", "models", "best_h_denseunet_model.pt")
    base_loaded = torch.load(base_ckpt_path, map_location=device)
    # handle dict or raw state
    if isinstance(base_loaded, dict) and 'model_state' in base_loaded:
        base_model.load_state_dict(base_loaded['model_state'])
    elif isinstance(base_loaded, dict) and 'model' in base_loaded:
        base_model.load_state_dict(base_loaded['model'])
    else:
        try:
            base_model.load_state_dict(base_loaded)
        except Exception:
            # fallback: try direct load_state (some checkpoints save raw state_dict under different keys)
            base_model.load_state_dict({k.replace('module.', ''):v for k,v in base_loaded.items()})
    base_results = evaluate_model(base_model, dataset, device, patch_size=(16, 128, 128))
    # save baseline metrics CSV
    csv_path_base = os.path.join(results_dir, "baseline_metrics.csv")
    with open(csv_path_base, "w", newline="") as f:
        writer = csv.writer(f)
        if len(base_results)>0:
            keys = ["case"] + [k for k in base_results[0].keys() if k != "case"]
            writer.writerow(["model"] + keys)
            for r in base_results:
                writer.writerow(["baseline"] + [r[k] for k in keys])
    del base_model; torch.cuda.empty_cache()

    # ---------------- Stage 2: joint ----------------
    print("[Eval] Stage 2: Evaluating joint model...")
    joint_model = HybridHDenseUNet(encoder_out_ch=64, n_classes=3).to(device)
    joint_model.n_classes = 3
    joint_ckpt_path = os.path.join("..", "models", "joint_seg_best.pt")
    joint_loaded = torch.load(joint_ckpt_path, map_location=device)
    if isinstance(joint_loaded, dict) and 'model' in joint_loaded:
        joint_model.load_state_dict(joint_loaded['model'])
    elif isinstance(joint_loaded, dict) and 'model_state' in joint_loaded:
        joint_model.load_state_dict(joint_loaded['model_state'])
    else:
        try:
            joint_model.load_state_dict(joint_loaded)
        except Exception:
            joint_model.load_state_dict({k.replace('module.', ''):v for k,v in joint_loaded.items()})
    joint_results = evaluate_model(joint_model, dataset, device, patch_size=(16, 128, 128))
    csv_path_joint = os.path.join(results_dir, "joint_metrics.csv")
    with open(csv_path_joint, "w", newline="") as f:
        writer = csv.writer(f)
        if len(joint_results)>0:
            keys = ["case"] + [k for k in joint_results[0].keys() if k != "case"]
            writer.writerow(["model"] + keys)
            for r in joint_results:
                writer.writerow(["joint"] + [r[k] for k in keys])
    del joint_model; torch.cuda.empty_cache()

    # ---- Merge metrics into one csv (paired if equal length) ----
    merged_csv = os.path.join(results_dir, "metrics.csv")
    if len(base_results) and len(joint_results):
        keys = ["case"] + [k for k in base_results[0].keys() if k != "case"]
        with open(merged_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model"] + keys)
            for br, jr in zip(base_results, joint_results):
                writer.writerow(["baseline"] + [br[k] for k in keys])
                writer.writerow(["joint"] + [jr[k] for k in keys])
        print(f"[Eval] Metrics saved → {merged_csv}")

    # ---- Summary ----
    def avg_std(results):
        out = {}
        if not results:
            return out
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
        f.write(f"Source-only Dice: {avg_base.get('dice_cls1', 0.0):.4f}\n")
        f.write(f"Joint-adapted Dice: {avg_joint.get('dice_cls1', 0.0):.4f}\n")
        f.write(f"DGRR: {avg_joint.get('dice_cls1',0.0)-avg_base.get('dice_cls1',0.0):.4f}\n\n")
        for k in sorted(set(list(avg_base.keys()) + list(avg_joint.keys()))):
            b = avg_base.get(k, 0.0); j = avg_joint.get(k, 0.0)
            f.write(f"{k:<12} src={b:.4f}, joint={j:.4f}\n")
        f.write("============================\n")
    print(f"[Eval] Summary saved → {summary_txt}")

    # ---- Visualization ----
    if len(base_results) and len(joint_results):
        plot_boxplot(base_results, joint_results, results_dir)
        plot_radar(avg_base, avg_joint, results_dir)
        print("[Eval] Visualizations saved!")

    print("[Eval] All done.")

if __name__ == "__main__":
    main()
