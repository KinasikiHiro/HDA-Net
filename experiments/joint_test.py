"""
train_joint_test.py
-------------------------------------
Joint training test: Segmentation (HybridHDenseUNet) + generated labels
- Single epoch
- Mixed source + generated batch
- Automatic mixed precision for reduced GPU memory
- Compatible with both 2D and 3D generated data
"""

import os, sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import h5py

# Path setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
UTILS_DIR = os.path.join(ROOT_DIR, "utils")
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

# Import your H5Dataset / collate_fn
try:
    from io_utils import H5Dataset, collate_fn
except:
    H5Dataset = None
    collate_fn = None

# Import segmentation model
sys.path.append(os.path.join(ROOT_DIR, "experiments"))
from train_h_denseunet import HybridHDenseUNet


# -------------------------
# Simple IO utils
# -------------------------
def load_h5_simple(path):
    with h5py.File(path, "r") as f:
        return {k: f[k][()] for k in f.keys()}


# -------------------------
# Slice-wise GeneratedVolumeDataset
# -------------------------
class GeneratedVolumeDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".h5")])
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
            if label_vol.ndim == 3:
                label_slice = label_vol[:, :, z][None, None].astype(np.int16)  # (1,1,H,W)
            else:
                label_slice = label_vol[None, None].astype(np.int16)

            if 'image' in f:
                img_vol = f['image'][()]
                if img_vol.ndim == 3:  # (H,W,D)
                    img_slice = img_vol[:, :, z][None, None].astype(np.float32)
                elif img_vol.ndim == 4:  # (C,H,W,D)
                    img_slice = img_vol[:, :, :, z][None].astype(np.float32)
                else:  # (H,W)
                    img_slice = img_vol[None, None].astype(np.float32)
            else:
                img_slice = np.zeros_like(label_slice, dtype=np.float32)

        img_t = torch.from_numpy(img_slice).float()
        lbl_t = torch.from_numpy(label_slice).long()
        return img_t, lbl_t


# -------------------------
# Loss functions
# -------------------------
def dice_loss(pred, target, eps=1e-6):
    probs = F.softmax(pred, dim=1)
    B, C = probs.shape[:2]
    dims = tuple(range(2, probs.ndim))
    target_onehot = torch.zeros_like(probs)
    target_onehot.scatter_(1, target.unsqueeze(1), 1)
    inter = (probs * target_onehot).sum(dim=dims)
    card = probs.sum(dim=dims) + target_onehot.sum(dim=dims)
    dice = (2 * inter + eps) / (card + eps)
    return 1 - dice.mean()


# -------------------------
# Single epoch joint test
# -------------------------
def train_joint_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[JointTest] Device:", device)

    # Generated data
    generated_dir = os.path.abspath(args.generated_dir)
    if not os.path.exists(generated_dir) or len(os.listdir(generated_dir)) == 0:
        raise RuntimeError(f"[JointTest] generated_dir empty: {generated_dir}")
    else:
        print(f"[JointTest] Found pre-generated diffusion outputs under {generated_dir}, skip sampling.")

    # Source dataset
    src_dir = os.path.join(ROOT_DIR, "preprocessed", "train")
    if H5Dataset is not None:
        src_ds = H5Dataset(src_dir, crop_size=(args.crop_d, args.crop_h, args.crop_w), augment=False)
        src_loader = DataLoader(src_ds, batch_size=args.src_batch, shuffle=True,
                                num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    else:
        raise RuntimeError("H5Dataset not found")

    # Generated dataset (slice-wise)
    gen_ds = GeneratedVolumeDataset(generated_dir)
    gen_loader = DataLoader(gen_ds, batch_size=args.gen_batch, shuffle=True,
                            num_workers=max(0, args.num_workers // 2), pin_memory=True)

    # Model
    seg_net = HybridHDenseUNet(encoder_out_ch=args.enc_ch, n_classes=args.num_classes, pretrained_encoder=False).to(device)
    optimizer = optim.Adam(seg_net.parameters(), lr=args.lr_seg)
    ce_loss = nn.CrossEntropyLoss()

    # Mixed precision
    scaler = torch.amp.GradScaler()

    seg_net.train()
    gen_iter = iter(gen_loader)
    total_loss = 0.0
    n_steps = 0

    for src_batch in tqdm(src_loader, desc="[JointTest] Single epoch"):
        imgs_src, labels_src = src_batch
        imgs_src, labels_src = imgs_src.to(device), labels_src.to(device)

        # Fetch generated batch
        try:
            imgs_gen, labels_gen = next(gen_iter)
        except StopIteration:
            gen_iter = iter(gen_loader)
            imgs_gen, labels_gen = next(gen_iter)
        imgs_gen, labels_gen = imgs_gen.to(device), labels_gen.to(device)

        # Auto-select interpolation mode
        if imgs_gen.ndim == 5:  # (B,C,D,H,W)
            imgs_gen = F.interpolate(imgs_gen, size=(args.crop_d, args.crop_h, args.crop_w),
                                     mode='trilinear', align_corners=False)
            labels_gen = F.interpolate(labels_gen.float(), size=(args.crop_d, args.crop_h, args.crop_w),
                                       mode='nearest').long()
        elif imgs_gen.ndim == 4:  # (B,C,H,W)
            imgs_gen = F.interpolate(imgs_gen, size=(args.crop_h, args.crop_w),
                                     mode='bilinear', align_corners=False)
            labels_gen = F.interpolate(labels_gen.float(), size=(args.crop_h, args.crop_w),
                                       mode='nearest').long()

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            out_src = seg_net(imgs_src)
            loss_src = args.alpha * ce_loss(out_src, labels_src.squeeze(1)) + (1 - args.alpha) * dice_loss(out_src, labels_src.squeeze(1))
            out_gen = seg_net(imgs_gen)
            loss_gen = args.alpha * ce_loss(out_gen, labels_gen.squeeze(1)) + (1 - args.alpha) * dice_loss(out_gen, labels_gen.squeeze(1))
            loss = loss_src + args.lambda_gen * loss_gen

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        n_steps += 1

    print(f"[JointTest] Single epoch finished, Avg Loss: {total_loss / max(1, n_steps):.5f}")


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_dir", type=str, default=r"..\preprocessed\generated_labels")
    parser.add_argument("--src_batch", type=int, default=1)
    parser.add_argument("--gen_batch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr_seg", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lambda_gen", type=float, default=1.0)
    parser.add_argument("--enc_ch", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--crop_d", type=int, default=32)
    parser.add_argument("--crop_h", type=int, default=128)
    parser.add_argument("--crop_w", type=int, default=128)
    args = parser.parse_args()

    train_joint_test(args)
