"""
pseudo_h_denseunet.py
-------------------------------------
Generate pseudo labels (and confidence maps) for target domain (settingB)
using a pretrained source-domain baseline model (H-DenseUNet).

Output:
    preprocessed/pseudo_labels/{case_name}.h5
Fields:
    image          (z, y, x)
    pseudo_label   (z, y, x)
    pseudo_prob    (z, y, x)
    entropy        (z, y, x)
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import h5py

# === Import utils ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
UTILS_DIR = os.path.join(ROOT_DIR, "utils")
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

from io_utils import H5Dataset, collate_fn, save_h5
from train_h_denseunet import HybridHDenseUNet  # reuse model

# ------------------------
# Generate pseudo labels
# ------------------------
@torch.no_grad()
def generate_pseudo_labels(model_path, input_dir, output_dir, batch_size=1, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Loading model from {model_path}")

    # === Load model ===
    model = HybridHDenseUNet(encoder_out_ch=64, n_classes=3, pretrained_encoder=False)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # === Load dataset ===
    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".h5")])
    print(f"Generating pseudo labels for {len(files)} test cases...")

    for f_name in tqdm(files):
        in_path = os.path.join(input_dir, f_name)
        out_path = os.path.join(output_dir, f_name)

        with h5py.File(in_path, "r") as f:
            image = f["image"][()]
        image = image.astype(np.float32)
        D, H, W = image.shape

        # Prepare input tensor: (1,1,D,H,W)
        img_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)

        # Forward in slices to avoid OOM
        max_slices = 8
        preds_all, probs_all, entropy_all = [], [], []
        for i in range(0, D, max_slices):
            chunk = img_t[:, :, i:i+max_slices]
            out = model(chunk)
            prob = F.softmax(out, dim=1)
            label = prob.argmax(dim=1)
            ent = -(prob * torch.log(prob + 1e-8)).sum(dim=1) / np.log(prob.shape[1])  # normalized entropy

            preds_all.append(label.cpu().numpy())
            probs_all.append(prob.max(dim=1)[0].cpu().numpy())
            entropy_all.append(ent.cpu().numpy())

        pred_full = np.concatenate(preds_all, axis=1)[0]      # (D,H,W)
        prob_full = np.concatenate(probs_all, axis=1)[0]      # (D,H,W)
        entropy_full = np.concatenate(entropy_all, axis=1)[0]  # (D,H,W)

        # === Save combined result ===
        save_h5(out_path, image=image, label=pred_full)
        with h5py.File(out_path, "a") as f:
            f.create_dataset("pseudo_prob", data=prob_full, compression="gzip")
            f.create_dataset("entropy", data=entropy_full, compression="gzip")

    print("Pseudo labels (with confidence & entropy) generation completed.")


# ------------------------
# Main entry
# ------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate pseudo labels (with confidence) for target domain")
    parser.add_argument("--model_path", type=str, default=os.path.join(ROOT_DIR, "models", "best_h_denseunet_model.pt"))
    parser.add_argument("--input_dir", type=str, default=os.path.join(ROOT_DIR, "preprocessed", "test"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(ROOT_DIR, "preprocessed", "pseudo_labels"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    generate_pseudo_labels(args.model_path, args.input_dir, args.output_dir, args.batch_size, args.device)
