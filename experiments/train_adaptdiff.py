"""
sample_adaptdiff.py
-------------------------------------
Fast sampling script for AdaptDiff-like conditional diffusion model.

Optimizations:
1. Batch processing
2. Half-precision (AMP) inference
3. Proper channel matching with training
"""

import os
import sys
import torch
import numpy as np
import h5py
from tqdm import tqdm
from skimage.transform import resize

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(os.path.join(ROOT_DIR, "utils"))

# ------------------------
# IO utilities
# ------------------------
def load_h5(path):
    with h5py.File(path, "r") as f:
        return {k: f[k][()] for k in f.keys()}

def save_h5(path, data_dict):
    with h5py.File(path, "w") as f:
        for k, v in data_dict.items():
            f.create_dataset(k, data=v, compression="gzip")

# ------------------------
# Model definitions
# ------------------------
class SimpleUNet(torch.nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()
        self.enc1 = torch.nn.Sequential(torch.nn.Conv2d(in_ch, base_ch, 3, padding=1), torch.nn.ReLU())
        self.enc2 = torch.nn.Sequential(torch.nn.Conv2d(base_ch, base_ch*2, 3, padding=1), torch.nn.ReLU())
        self.dec1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(base_ch*2, base_ch, 3, padding=1), torch.nn.ReLU())
        self.outc = torch.nn.Conv2d(base_ch, out_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = t.view(-1,1,1,1)
        x = x + t_emb
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.dec1(x)
        x = self.outc(x)
        return x

class DiffusionModel(torch.nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    def q_sample(self, x0, t, noise):
        alphas_cumprod = self.alphas_cumprod.to(x0.device)
        sqrt_acp = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
        sqrt_omp = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]
        return sqrt_acp * x0 + sqrt_omp * noise

    @torch.no_grad()
    def p_sample(self, x_t, t):
        t_tensor = torch.tensor([t/float(self.timesteps)], device=x_t.device)
        pred_noise = self.model(x_t, t_tensor)
        alpha_t = self.alphas_cumprod[t]
        x_prev = (x_t - (1-alpha_t).sqrt()*pred_noise)/alpha_t.sqrt()
        return x_prev

# ------------------------
# Dataset
# ------------------------
class AdaptDiffDataset(torch.utils.data.Dataset):
    def __init__(self, folder, mode="label2image", target_size=(256,256)):
        self.files = sorted([os.path.join(folder,f) for f in os.listdir(folder) if f.endswith(".h5")])
        self.mode = mode
        self.target_size = target_size
        self.slices = []
        for fidx, fpath in enumerate(self.files):
            with h5py.File(fpath,"r") as f:
                n_slices = f["image"].shape[2]
                for z in range(n_slices):
                    self.slices.append((fidx,z))

    def __len__(self): return len(self.slices)

    def __getitem__(self, idx):
        fidx,z = self.slices[idx]
        path = self.files[fidx]
        with h5py.File(path,"r") as f:
            img = resize(f["image"][:,:,z], self.target_size, anti_aliasing=True).astype(np.float32)
            lbl = resize(f["label"][:,:,z], self.target_size, anti_aliasing=True).astype(np.float32)
            prob = resize(f["pseudo_prob"][:,:,z], self.target_size, anti_aliasing=True).astype(np.float32) if "pseudo_prob" in f else np.zeros_like(lbl, dtype=np.float32)
        img = torch.from_numpy(img).unsqueeze(0)
        lbl = torch.from_numpy(lbl).unsqueeze(0)
        prob = torch.from_numpy(prob).unsqueeze(0)

        # 保持与训练一致的通道
        if self.mode=="label2image":
            cond = torch.cat([lbl, torch.zeros_like(lbl)], dim=0)  # 2通道
        else:
            cond = torch.cat([img, prob, torch.zeros_like(img)], dim=0)  # 3通道
        return cond, (img, lbl)

# ------------------------
# Sampling loop
# ------------------------
@torch.no_grad()
def sample_adaptdiff(ckpt_path, input_dir, output_dir, mode="label2image", num_steps=100, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # 通道匹配
    in_ch = 2 if mode=="label2image" else 3
    out_ch = 1

    # 模型加载
    unet = SimpleUNet(in_ch=in_ch, out_ch=out_ch)
    diffusion = DiffusionModel(unet).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    diffusion.load_state_dict(state_dict)
    diffusion.eval()

    # 数据加载
    dataset = AdaptDiffDataset(input_dir, mode=mode)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    for cond_batch, (img_batch, lbl_batch) in tqdm(loader):
        cond_batch = cond_batch.to(device)
        x_t = torch.randn_like(cond_batch)

        # 使用 AMP 加速
        for t in reversed(range(num_steps)):
            with torch.cuda.amp.autocast():
                x_t = diffusion.p_sample(x_t, t)

        out_batch = x_t.cpu().numpy()
        # 保存每个样本
        for i in range(out_batch.shape[0]):
            out = np.clip(out_batch[i,0], 0, 1)
            fname = os.path.basename(dataset.files[dataset.slices[i][0]])
            out_name = fname.replace(".h5", f"_gen_{i}.h5")
            save_path = os.path.join(output_dir, out_name)
            if mode=="label2image":
                save_h5(save_path, {"image": out.astype(np.float32)})
            else:
                label_hard = (out>0.5).astype(np.uint8)
                save_h5(save_path, {"label": label_hard})

# ------------------------
# CLI
# ------------------------
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fast sampling for AdaptDiff model")
    parser.add_argument("--ckpt", type=str, default=r"..\models\adaptdiff_image2label.pt")
    parser.add_argument("--input_dir", type=str, default=r"..\preprocessed\pseudo_labels")
    parser.add_argument("--output_dir", type=str, default=r"..\preprocessed\generated_labels")
    parser.add_argument("--mode", type=str, choices=["label2image","image2label"], default="image2label")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    sample_adaptdiff(args.ckpt, args.input_dir, args.output_dir, args.mode, args.num_steps, args.batch_size)