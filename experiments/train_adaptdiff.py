"""
train_adaptdiff_volume_resume.py
-------------------------------------
3D Volume-wise chunked training for image2label diffusion model (AdaptDiff-like)
支持断点续训与最优模型保存。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from tqdm import tqdm

from train_h_denseunet import HybridHDenseUNet


# ----------------------------
# Dataset: volume-wise
# ----------------------------
class AdaptDiffVolumeDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".h5")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with h5py.File(path, "r") as f:
            img = f["image"][()].astype(np.float32)
            lbl = f["label"][()].astype(np.float32) if "label" in f else np.zeros_like(img, dtype=np.float32)
            prob = f["pseudo_prob"][()].astype(np.float32) if "pseudo_prob" in f else np.zeros_like(img, dtype=np.float32)

        # H,W,D -> D,H,W
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        prob = torch.from_numpy(prob).permute(2, 0, 1).unsqueeze(0)
        lbl = torch.from_numpy(lbl).permute(2, 0, 1).unsqueeze(0)

        cond = torch.cat([img, prob], dim=0)  # [C=2, D, H, W]
        return cond, lbl


# ----------------------------
# 3D UNet backbone
# ----------------------------
class SimpleUNet3D(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base_ch=32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv3d(in_ch, base_ch, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv3d(base_ch, base_ch * 2, 3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose3d(base_ch * 2, base_ch, 3, padding=1), nn.ReLU())
        self.outc = nn.Conv3d(base_ch, out_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = t.view(-1, 1, 1, 1, 1)
        x = x + t_emb
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.dec1(x)
        x = self.outc(x)
        return x


# ----------------------------
# 3D Diffusion
# ----------------------------
class DiffusionModel3D(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    def q_sample(self, x0, t, noise):
        alphas_cumprod = self.alphas_cumprod.to(x0.device)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])[:, None, None, None, None]
        sqrt_one_minus = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None, None]
        return sqrt_alphas_cumprod * x0 + sqrt_one_minus * noise

    def forward(self, x0, cond):
        x0 = x0.float()
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        input_model = torch.cat([xt, cond], dim=1)
        pred_noise = self.model(input_model, t.float() / self.timesteps)
        return F.mse_loss(pred_noise, noise)


# ----------------------------
# Edge loss (3D)
# ----------------------------
def edge_loss_3d(pred, target):
    device = pred.device
    sobel_x = torch.zeros((1, 1, 3, 3, 3), device=device)
    sobel_y = torch.zeros((1, 1, 3, 3, 3), device=device)
    sobel_z = torch.zeros((1, 1, 3, 3, 3), device=device)

    sobel_x[0, 0, 1, :, :] = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=device)
    sobel_y[0, 0, 1, :, :] = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=device)
    sobel_z[0, 0, :, 1, 1] = torch.tensor([1, 0, -1], device=device)

    pred_gx = F.conv3d(pred, sobel_x, padding=1)
    pred_gy = F.conv3d(pred, sobel_y, padding=1)
    pred_gz = F.conv3d(pred, sobel_z, padding=1)

    targ_gx = F.conv3d(target, sobel_x, padding=1)
    targ_gy = F.conv3d(target, sobel_y, padding=1)
    targ_gz = F.conv3d(target, sobel_z, padding=1)

    return F.mse_loss(pred_gx, targ_gx) + F.mse_loss(pred_gy, targ_gy) + F.mse_loss(pred_gz, targ_gz)


# ----------------------------
# Training loop
# ----------------------------
def train_adaptdiff_volume(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AdaptDiffVolumeDataset(os.path.join(args.root_dir, "preprocessed/pseudo_labels"))
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    unet = SimpleUNet3D(in_ch=3, out_ch=1).to(device)
    diffusion = DiffusionModel3D(unet).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler()

    # 路径定义
    ckpt_path = os.path.join(args.root_dir, "models/adaptdiff_image2label_3d.pt")
    best_path = os.path.join(args.root_dir, "models/best_adaptdiff_image2label_3d.pt")

    # 加载已有模型（断点续训）
    start_epoch = 0
    best_loss = float("inf")
    if os.path.exists(ckpt_path):
        print(f"🔄 Resuming training from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        # ✅ 自动判断文件结构
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            diffusion.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_loss = checkpoint.get("best_loss", float("inf"))
            print(f"Resumed full checkpoint from epoch {start_epoch}, best loss {best_loss:.6f}")
        else:
            # ✅ 兼容旧版，仅含模型参数
            diffusion.load_state_dict(checkpoint)
            print("⚠️ Loaded old model weights only (no optimizer/scaler). Starting fresh training state.")

    # Teacher model
    teacher_ckpt = os.path.join(args.root_dir, "models/best_h_denseunet_model.pt")
    if os.path.exists(teacher_ckpt):
        teacher = HybridHDenseUNet(encoder_out_ch=64, n_classes=3, pretrained_encoder=False).to(device)
        checkpoint = torch.load(teacher_ckpt, map_location=device)
        teacher.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint)
        teacher.eval()
        print("✅ Teacher loaded.")
    else:
        teacher = None
        print("⚠️ Teacher not found, semantic loss skipped.")

    max_slices = 16

    for epoch in range(start_epoch, args.epochs):
        diffusion.train()
        epoch_loss = 0
        for cond, target in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            cond, target = cond.to(device).float(), target.to(device).float()
            B, C, D, H, W = cond.shape

            for start in range(0, D, max_slices):
                end = min(start + max_slices, D)
                cond_chunk = cond[:, :, start:end, :, :]
                target_chunk = target[:, :, start:end, :, :]

                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda'):
                    loss_diff = diffusion(target_chunk, cond_chunk)
                    loss_total = loss_diff

                    if teacher is not None:
                        with torch.no_grad():
                            pred_teacher = teacher(target_chunk)
                            pred_teacher = F.softmax(pred_teacher, dim=1)
                        target_onehot = torch.zeros_like(pred_teacher)
                        target_indices = target_chunk.long()
                        target_onehot.scatter_(1, target_indices, 1.0)
                        loss_sem = F.mse_loss(target_onehot, pred_teacher)
                        loss_total += args.lambda_sem * loss_sem

                    loss_total += args.lambda_edge * edge_loss_3d(target_chunk, cond_chunk[:, :1])

                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss_total.item()

        avg_epoch_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} finished, avg loss: {avg_epoch_loss:.6f}")

        # 保存 checkpoint
        torch.save({
            "epoch": epoch,
            "model": diffusion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_loss": best_loss,
        }, ckpt_path)
        print(f"💾 Checkpoint saved to {ckpt_path}")

        # 保存最佳模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(diffusion.state_dict(), best_path)
            print(f"🏆 Best model updated (loss={best_loss:.6f}) -> {best_path}")


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=r"..")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_sem", type=float, default=0.1)
    parser.add_argument("--lambda_edge", type=float, default=0.05)
    args = parser.parse_args()

    train_adaptdiff_volume(args)
