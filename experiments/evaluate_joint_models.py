import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import csv
from collections import defaultdict
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# ------------------------- Path setup -------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(os.path.join(ROOT_DIR, "experiments"))

from train_h_denseunet import HybridHDenseUNet

# ------------------------- Dataset -------------------------
class PseudoLabelSliceDataset(Dataset):
    def __init__(self, folder, case_name=None):
        self.files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".h5")])
        if case_name:
            self.files = [f for f in self.files if case_name in f]
            if len(self.files) == 0:
                raise RuntimeError(f"[Dataset] No file matches case_name={case_name}")
        self.slice_index = []
        for fidx, fpath in enumerate(self.files):
            with h5py.File(fpath, "r") as f:
                D = f['label'].shape[2] if f['label'].ndim == 3 else 1
                for z in range(D):
                    self.slice_index.append((fidx, z))

    def __len__(self):
        return len(self.slice_index)

    def __getitem__(self, idx):
        fidx, z = self.slice_index[idx]
        with h5py.File(self.files[fidx], "r") as f:
            label_vol = f['label'][()]
            label_slice = label_vol[:, :, z][None, None].astype(np.int16) if label_vol.ndim==3 else label_vol[None, None].astype(np.int16)
            img_vol = f['image'][()] if 'image' in f else np.zeros(label_vol.shape, dtype=np.float32)
            img_slice = img_vol[:, :, z][None, None].astype(np.float32) if img_vol.ndim==3 else img_vol[None, None].astype(np.float32)
        return torch.from_numpy(img_slice).float(), torch.from_numpy(label_slice).long(), os.path.basename(self.files[fidx]), z


# ------------------------- Metrics -------------------------
EPS = 1e-6
def dice(pred, target):
    inter = (pred*target).sum()
    return ((2*inter+EPS)/(pred.sum()+target.sum()+EPS)).item()
def iou(pred, target):
    inter = (pred*target).sum()
    union = pred.sum()+target.sum()-inter
    return ((inter+EPS)/(union+EPS)).item()
def precision(pred, target):
    tp = (pred*target).sum()
    fp = (pred*(1-target)).sum()
    return ((tp+EPS)/(tp+fp+EPS)).item()
def recall(pred, target):
    tp = (pred*target).sum()
    fn = ((1-pred)*target).sum()
    return ((tp+EPS)/(tp+fn+EPS)).item()
def volume_diff(pred, target):
    return abs(pred.sum().item() - target.sum().item()) / (target.sum().item()+EPS)

# ------------------------- Slice-wise Evaluation -------------------------
def evaluate_model_slicewise(model, dataset, device, n_classes=3, vis_dir=None, case_idx=-1, batch_size=1):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    per_volume_metrics = defaultdict(list)
    conf_values_per_volume = defaultdict(list)

    with torch.no_grad():
        for img, label, fname, z in tqdm(loader, desc="Evaluating (slice-wise)"):
            if case_idx >=0 and fname != dataset.files[case_idx]:
                continue

            img = img.to(device)  # [1,1,H,W] or [1,C,H,W]
            label = label.to(device)

            # If single slice, add depth dimension
            if img.ndim == 4:
                img = img.unsqueeze(2)  # [B,C,D,H,W]

            # AMP autocast updated
            with torch.amp.autocast(device_type='cuda'):
                out = model(img)  # [B,C,D,H,W]

            # Resize prediction to label size
            out = F.interpolate(out, size=label.shape[2:], mode='trilinear', align_corners=False)

            if n_classes == 1:
                probs = torch.sigmoid(out)
                pred = (probs>0.5).float()
                conf_map = probs.cpu().squeeze().numpy()
            else:
                probs = F.softmax(out, dim=1)
                pred_labels = probs.argmax(dim=1, keepdim=True)
                conf_map, _ = probs.max(dim=1)
                pred = (pred_labels>0).float()
                conf_map = conf_map.cpu().squeeze().numpy()

            # Ensure pred and target are on CPU for metric computation
            tgt_fg = (label.cpu()>0).float()
            pred_fg = pred.cpu()

            m = {
                "dice": dice(pred_fg, tgt_fg),
                "iou": iou(pred_fg, tgt_fg),
                "precision": precision(pred_fg, tgt_fg),
                "recall": recall(pred_fg, tgt_fg),
                "vol_diff": volume_diff(pred_fg, tgt_fg),
                "conf_mean": conf_map.mean(),
                "conf_ratio": (conf_map>0.9).mean()
            }

            per_volume_metrics[fname].append(m)
            conf_values_per_volume[fname].append(conf_map)

            # visualize a single slice if needed
            if vis_dir and case_idx>=0:
                os.makedirs(vis_dir, exist_ok=True)
                mid_slice = img.shape[2]//2
                fig, axs = plt.subplots(1,4,figsize=(14,4))
                axs[0].imshow(img.cpu().squeeze()[mid_slice], cmap='gray'); axs[0].set_title("Input")
                axs[1].imshow(label.cpu().squeeze(), cmap='gray'); axs[1].set_title("GT")
                axs[2].imshow(pred.cpu().squeeze(), cmap='gray'); axs[2].set_title("Pred")
                im = axs[3].imshow(conf_map[mid_slice], cmap='jet'); axs[3].set_title("Confidence")
                for ax in axs: ax.axis('off')
                fig.colorbar(im, ax=axs[3], fraction=0.046, pad=0.02)
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir,f"{fname}_slice{z}_vis.png"))
                plt.close()

    # Aggregate metrics per volume
    agg_metrics = {}
    for fname, metrics_list in per_volume_metrics.items():
        agg = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
        agg_metrics[fname] = agg

    # Compute overall average
    overall_agg = {k: np.mean([agg_metrics[f][k] for f in agg_metrics]) for k in list(metrics_list[0].keys())}

    return overall_agg, agg_metrics, conf_values_per_volume

# ------------------------- Main -------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_model", type=str, default=os.path.join("..", "models", "best_h_denseunet_model.pt"))
    parser.add_argument("--joint_model", type=str, default=os.path.join("..", "models", "joint_seg_best.pt"))
    parser.add_argument("--pseudo_dir", type=str, default=os.path.join("..", "preprocessed", "pseudo_labels"))
    parser.add_argument("--result_dir", type=str, default=os.path.join("..", "results", "joint_evaluate"))
    parser.add_argument("--case_idx", type=int, default=-1)
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--n_classes", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.result_dir, exist_ok=True)

    dataset = PseudoLabelSliceDataset(args.pseudo_dir, case_name=args.case_name)

    model_src = HybridHDenseUNet(n_classes=args.n_classes).to(device)
    model_joint = HybridHDenseUNet(n_classes=args.n_classes).to(device)

    state_src = torch.load(args.src_model, map_location=device)
    model_src.load_state_dict(state_src if isinstance(state_src, dict) else state_src)

    state_joint = torch.load(args.joint_model, map_location=device)
    if "model" in state_joint: state_joint = state_joint["model"]
    model_joint.load_state_dict(state_joint)

    print("[Eval] Evaluating Source-only model...")
    agg_src, per_volume_src, conf_src = evaluate_model_slicewise(model_src, dataset, device, args.n_classes, batch_size=args.batch_size)
    print("[Eval] Evaluating Joint-adapted model...")
    agg_joint, per_volume_joint, conf_joint = evaluate_model_slicewise(model_joint, dataset, device, args.n_classes, batch_size=args.batch_size)

    DGRR = (agg_joint["dice"] - agg_src["dice"]) / (1 - agg_src["dice"] + EPS)

    # Save CSV
    def save_csv(metrics_dict, path):
        keys = ["file","dice","iou","precision","recall","vol_diff","conf_mean","conf_ratio"]
        with open(path,"w",newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for fname, m in metrics_dict.items():
                row = {"file": fname}
                row.update(m)
                writer.writerow(row)

    save_csv(per_volume_src, os.path.join(args.result_dir,"eval_source.csv"))
    save_csv(per_volume_joint, os.path.join(args.result_dir,"eval_joint.csv"))

    # Save summary
    summary_file = os.path.join(args.result_dir,"eval_summary.txt")
    with open(summary_file,"w") as f:
        f.write("==== Evaluation Summary ====\n")
        f.write(f"Source-only Dice: {agg_src['dice']:.4f}\n")
        f.write(f"Joint-adapted Dice: {agg_joint['dice']:.4f}\n")
        f.write(f"DGRR: {DGRR:.4f}\n")
        for k in ["dice","iou","precision","recall","vol_diff","conf_mean","conf_ratio"]:
            f.write(f"{k:<12} src={agg_src[k]:.4f}, joint={agg_joint[k]:.4f}\n")
        f.write("============================\n")
    print(f"[Eval] Done. Summary saved to {summary_file}")
    print("DGRR =", DGRR)

if __name__ == "__main__":
    main()
