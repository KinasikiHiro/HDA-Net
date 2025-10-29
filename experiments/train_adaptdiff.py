"""
train_adaptdiff.py
-------------------------------------
Conditional diffusion model training for domain adaptation (AdaptDiff-like).

Modes:
  (A) label2image  - learn to generate target-style image given pseudo label.
  (B) image2label  - learn to refine pseudo label given image + prob.

Input:
  preprocessed/pseudo_labels/*.h5
Each file must contain:
  'image', 'label', 'pseudo_prob', 'entropy'

Output:
  models/adaptdiff_label2image.pt or models/adaptdiff_image2label.pt
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from tqdm import tqdm

# === Path setup ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
UTILS_DIR = os.path.join(ROOT_DIR, "utils")
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "experiments")
if EXPERIMENTS_DIR not in sys.path:
    sys.path.append(EXPERIMENTS_DIR)

# === Import teacher model ===
from train_h_denseunet import HybridHDenseUNet

# ------------------------------------------------
# Dataset loader
# ------------------------------------------------
class AdaptDiffDataset(Dataset):
    def __init__(self, folder, mode="label2image", target_size=(256,256)):
        self.files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".h5")])
        self.mode = mode
        self.target_size = target_size
        self.slices = []

        print("Setting up slices for delayed loading...")
        for fidx, fpath in enumerate(self.files):
            with h5py.File(fpath, "r") as f:
                n_slices = f["image"].shape[2]
                for z in range(n_slices):
                    self.slices.append((fidx, z))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        from skimage.transform import resize
        fidx, z = self.slices[idx]
        path = self.files[fidx]
        with h5py.File(path, "r") as f:
            img = resize(f["image"][:,:,z], self.target_size, anti_aliasing=True).astype(np.float32)
            lbl = resize(f["label"][:,:,z], self.target_size, anti_aliasing=True).astype(np.float32)
            prob = resize(f["pseudo_prob"][:,:,z], self.target_size, anti_aliasing=True).astype(np.float32) \
                   if "pseudo_prob" in f else np.zeros_like(lbl, dtype=np.float32)

        img = torch.from_numpy(img).unsqueeze(0)
        lbl = torch.from_numpy(lbl).unsqueeze(0)
        prob = torch.from_numpy(prob).unsqueeze(0)

        if self.mode == "label2image":
            cond = lbl
            target = img
        else:
            cond = torch.cat([img, prob], dim=0)
            target = lbl
        return cond, target

# ------------------------------------------------
# Simple UNet backbone for diffusion
# ------------------------------------------------
class SimpleUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, base_ch * 2, 3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(base_ch * 2, base_ch, 3, padding=1), nn.ReLU())
        self.outc = nn.Conv2d(base_ch, out_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = t.view(-1,1,1,1)
        x = x + t_emb
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.dec1(x)
        x = self.outc(x)
        return x

# ------------------------------------------------
# Diffusion utilities
# ------------------------------------------------
class DiffusionModel(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    def q_sample(self, x0, t, noise):
        alphas_cumprod = self.alphas_cumprod.to(x0.device)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]
        return sqrt_alphas_cumprod * x0 + sqrt_one_minus * noise

    def forward(self, x0, cond):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred_noise = self.model(torch.cat([xt, cond], dim=1), t.float() / self.timesteps)
        return F.mse_loss(pred_noise, noise)

# ------------------------------------------------
# Edge loss (structure preservation)
# ------------------------------------------------
def edge_loss(pred, target):
    """Encourage boundary consistency using Sobel gradients."""
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=pred.device).view(1,1,3,3)
    sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=pred.device).view(1,1,3,3)
    pred_gx = F.conv2d(pred, sobel_x, padding=1)
    pred_gy = F.conv2d(pred, sobel_y, padding=1)
    targ_gx = F.conv2d(target, sobel_x, padding=1)
    targ_gy = F.conv2d(target, sobel_y, padding=1)
    return F.mse_loss(pred_gx, targ_gx) + F.mse_loss(pred_gy, targ_gy)

# ------------------------------------------------
# Training loop
# ------------------------------------------------
def train_adaptdiff(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load dataset ===
    dataset = AdaptDiffDataset(os.path.join(ROOT_DIR, "preprocessed", "pseudo_labels"), mode=args.mode)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    # === Build diffusion model ===
    if args.mode == "label2image":
        in_ch = 2
        out_ch = 1
    else:
        in_ch = 3
        out_ch = 1

    unet = SimpleUNet(in_ch=in_ch, out_ch=out_ch)
    diffusion = DiffusionModel(unet).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.lr)

    # === Load teacher model ===
    teacher = HybridHDenseUNet(encoder_out_ch=64, n_classes=3, pretrained_encoder=False).to(device)
    teacher_ckpt = os.path.join(ROOT_DIR, "models", "best_h_denseunet_model.pt")
    if os.path.exists(teacher_ckpt):
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
        teacher.eval()
        print("✅ Loaded teacher model successfully.")
    else:
        print("⚠️ Warning: Teacher checkpoint not found. Semantic loss will be skipped.")
        teacher = None

    # === Training ===
    scaler = torch.cuda.amp.GradScaler()
    os.makedirs(os.path.join(ROOT_DIR, "models"), exist_ok=True)
    save_path = os.path.join(ROOT_DIR, "models", f"adaptdiff_{args.mode}.pt")

    for epoch in range(args.epochs):
        diffusion.train()
        epoch_loss = 0
        for cond, target in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            cond, target = cond.to(device), target.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss_diff = diffusion(target, cond)
                loss_total = loss_diff

                # --- Semantic consistency loss ---
                if teacher is not None and args.mode == "image2label":
                    with torch.no_grad():
                        pred_teacher = teacher(target.unsqueeze(1).repeat(1,1,8,1,1))  # fake depth=8
                        pred_teacher = F.softmax(pred_teacher, dim=1).mean(dim=2).unsqueeze(1)
                    loss_sem = F.mse_loss(target, pred_teacher)
                    loss_total += args.lambda_sem * loss_sem

                # --- Edge structure loss ---
                loss_edge = edge_loss(target, cond[:, :1, :, :])
                loss_total += args.lambda_edge * loss_edge

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss_total.item()

        print(f"Epoch {epoch+1}: Avg loss = {epoch_loss/len(loader):.5f}")
        torch.save(diffusion.state_dict(), save_path)

    print(f"✅ Training complete. Model saved to {save_path}")

# ------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train AdaptDiff-like conditional diffusion model")
    parser.add_argument("--mode", type=str, default="image2label", choices=["label2image", "image2label"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_sem", type=float, default=0.1, help="Weight for semantic consistency loss")
    parser.add_argument("--lambda_edge", type=float, default=0.05, help="Weight for edge structure loss")
    args = parser.parse_args()
    train_adaptdiff(args)
