"""
sample_adaptdiff_image2label.py
---------------------------------
3D volume-wise sampling for image2label diffusion model (AdaptDiff-like)
Input:
  preprocessed/test/*.h5 (fields: 'image', 'label')
Output:
  preprocessed/generated_labels/case_XXX.h5 (fields: 'image', 'label')
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from tqdm import tqdm

# ----------------------------
# Dataset: volume-wise (ignore original label)
# ----------------------------
class TestVolumeDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".h5")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with h5py.File(path, "r") as f:
            img = f["image"][()].astype(np.float32)

        # H,W,D -> D,H,W
        img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)   # [1,D,H,W]
        return img

# ----------------------------
# 3D UNet backbone for diffusion
# ----------------------------
class SimpleUNet3D(nn.Module):
    def __init__(self, in_ch=1+1, out_ch=1, base_ch=32):  # 输入 channels: image + pseudo_prob=1
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv3d(in_ch, base_ch, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv3d(base_ch, base_ch*2, 3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose3d(base_ch*2, base_ch, 3, padding=1), nn.ReLU())
        self.outc = nn.Conv3d(base_ch, out_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = t.view(-1,1,1,1,1)
        x = x + t_emb
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.dec1(x)
        x = self.outc(x)
        return x

# ----------------------------
# 3D Diffusion model
# ----------------------------
class DiffusionModel3D(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

# ----------------------------
# Sampling function (volume-wise, chunked)
# ----------------------------
def sample_volume(diffusion, image, max_slices=16, device='cuda'):
    B, _, D, H, W = 1, 1, *image.shape[2:]  # batch=1
    sampled = torch.randn((B,1,D,H,W), device=device)  # 初始噪声
    pseudo_prob = torch.zeros_like(sampled)  # placeholder，image2label 输入2个channel

    cond = torch.cat([image, pseudo_prob], dim=1)  # [B,2,D,H,W]

    for start in range(0, D, max_slices):
        end = min(start + max_slices, D)
        cond_chunk = cond[:, :, start:end, :, :]
        x = sampled[:, :, start:end, :, :]

        # 反向扩散
        for t in reversed(range(diffusion.timesteps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.float32)
            input_model = torch.cat([x, cond_chunk], dim=1)
            with torch.no_grad():
                pred_noise = diffusion.model(input_model, t_batch / diffusion.timesteps)
                alpha_t = diffusion.alphas_cumprod[t]
                sqrt_alpha = torch.sqrt(alpha_t)
                sqrt_one_minus = torch.sqrt(1 - alpha_t)
                x = (x - sqrt_one_minus * pred_noise) / sqrt_alpha

        sampled[:, :, start:end, :, :] = x

    return sampled

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=r"..")
    parser.add_argument("--model_path", type=str, default=r"..\models\adaptdiff_image2label_3d.pt")
    parser.add_argument("--max_slices", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TestVolumeDataset(os.path.join(args.root_dir, "preprocessed/test"))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model
    unet = SimpleUNet3D(in_ch=2, out_ch=1).to(device)
    diffusion = DiffusionModel3D(unet).to(device)
    diffusion.model.load_state_dict(torch.load(args.model_path, map_location=device))
    diffusion.model.eval()

    save_folder = os.path.join(args.root_dir, "preprocessed/generated_labels")
    os.makedirs(save_folder, exist_ok=True)

    for idx, image in enumerate(tqdm(loader)):
        image = image.to(device)
        sampled_label = sample_volume(diffusion, image, max_slices=args.max_slices, device=device)

        out_file = os.path.join(save_folder, f"case_{idx:03d}.h5")
        with h5py.File(out_file, "w") as f:
            f.create_dataset("image", data=image.cpu().numpy())
            f.create_dataset("label", data=sampled_label.cpu().numpy())
