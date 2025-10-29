"""
train_h_denseunet.py
-------------------------------------
Train baseline H-DenseUNet-like model (Hybrid 2D→3D) on source domain (settingA).

Project directory assumption:
    HDA-Net/
      ├── preprocessed/
      │     ├── train/      # settingA (source)
      │     └── test/       # settingB (target)
      ├── utils/io.py
      ├── experiments/train_h_denseunet.py
      └── models/

Usage example:
    cd HDA-Net/experiments
    python train_h_denseunet.py --epochs 100 --batch_size 1 --pretrained
"""

import os
import sys
import time
import argparse
from tqdm import tqdm

# add utils to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
UTILS_DIR = os.path.join(ROOT_DIR, "utils")
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from io_utils import H5Dataset, collate_fn

# ------------------------
# Model definition
# ------------------------
class SliceEncoder2D(nn.Module):
    """2D encoder using DenseNet121 applied to each axial slice independently (supports 1-channel input)."""
    def __init__(self, out_channels=64, pretrained=True):
        super().__init__()
        # ---- Use new torchvision weights API ----
        try:
            from torchvision.models import DenseNet121_Weights
            dnet = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        except Exception:
            dnet = models.densenet121(pretrained=pretrained)

        # ---- Modify first conv layer for 1-channel input ----
        old_conv = dnet.features.conv0
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
        dnet.features.conv0 = new_conv

        # ---- Extract up to denseblock3 (instead of full 4) ----
        # This gives intermediate feature maps (256–512 channels depending on DenseNet)
        self.features = nn.Sequential(
            dnet.features.conv0,
            dnet.features.norm0,
            dnet.features.relu0,
            dnet.features.pool0,
            dnet.features.denseblock1,
            dnet.features.transition1,
            dnet.features.denseblock2,
            dnet.features.transition2,
            dnet.features.denseblock3
        )

        # ---- Dynamically detect output channels ----
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 256, 256)
            out = self.features(dummy)
            c_out = out.shape[1]  # typically 512 or 1024

        self.out_conv = nn.Conv2d(c_out, out_channels, kernel_size=1)

    def forward(self, x):
        f = self.features(x)
        f = self.out_conv(f)
        return f

class Simple3DDecoder(nn.Module):
    """3D decoder that fuses slice features."""
    def __init__(self, in_ch=64, n_classes=3):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(128)
        self.conv2 = nn.Conv3d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.up1 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        self.conv3 = nn.Conv3d(128, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.up2 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        self.conv4 = nn.Conv3d(64, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm3d(32)
        self.final = nn.Conv3d(32, n_classes, 1)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.up1(x)
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.up2(x)
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x = self.final(x)
        return x


class HybridHDenseUNet(nn.Module):
    """Hybrid 2D→3D segmentation model."""
    def __init__(self, encoder_out_ch=64, n_classes=3, pretrained_encoder=True):
        super().__init__()
        self.slice_encoder = SliceEncoder2D(out_channels=encoder_out_ch, pretrained=pretrained_encoder)
        self.decoder = Simple3DDecoder(in_ch=encoder_out_ch, n_classes=n_classes)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x2d = x.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
        feat2d = self.slice_encoder(x2d)
        _, F, Hp, Wp = feat2d.shape
        feat3d = feat2d.view(B, D, F, Hp, Wp).permute(0, 2, 1, 3, 4)
        out = self.decoder(feat3d)
        out = nn.functional.interpolate(out, size=(D, H, W), mode="trilinear", align_corners=False)
        return out


# ------------------------
# Losses
# ------------------------
def dice_loss(pred, target, eps=1e-6):
    probs = nn.functional.softmax(pred, dim=1)
    B, C, D, H, W = probs.shape
    target_onehot = torch.zeros((B, C, D, H, W), device=pred.device)
    target_onehot.scatter_(1, target.unsqueeze(1), 1)
    intersection = (probs * target_onehot).sum(dim=(2, 3, 4))
    cardinality = probs.sum(dim=(2, 3, 4)) + target_onehot.sum(dim=(2, 3, 4))
    dice = (2 * intersection + eps) / (cardinality + eps)
    return 1 - dice.mean()


# ------------------------
# Training loop
# ------------------------
def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # === Load dataset ===
    train_dir = os.path.join(ROOT_DIR, "preprocessed", "train")
    dataset = H5Dataset(train_dir, crop_size=(args.crop_d, args.crop_h, args.crop_w), augment=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # === Initialize model ===
    model = HybridHDenseUNet(encoder_out_ch=args.enc_ch, n_classes=args.num_classes, pretrained_encoder=args.pretrained)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.5)
    ce_loss = nn.CrossEntropyLoss()
    best_loss = float("inf")

    # === Prepare model dir ===
    model_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        for i, (imgs, labels) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss_ce = ce_loss(outputs, labels)
            loss_d = dice_loss(outputs, labels)
            loss = args.alpha * loss_ce + (1 - args.alpha) * loss_d
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        avg_loss = running_loss / len(loader)
        print(f"[Epoch {epoch+1}] Avg Loss = {avg_loss:.4f}, Time = {time.time()-t0:.1f}s")

        # === Save checkpoint ===
        ckpt_path = os.path.join(model_dir, f"hda_epoch{epoch+1:03d}.pt")
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "optim_state": optimizer.state_dict()}, ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(model_dir, "best_h_denseunet_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved to {best_path}")

    print(f"Training finished! Best loss = {best_loss:.4f}")


# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline H-DenseUNet (PyTorch)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_step", type=int, default=50)
    parser.add_argument("--crop_d", type=int, default=16)
    parser.add_argument("--crop_h", type=int, default=256)
    parser.add_argument("--crop_w", type=int, default=256)
    parser.add_argument("--enc_ch", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for CE vs Dice")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained encoder")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    train(args)
