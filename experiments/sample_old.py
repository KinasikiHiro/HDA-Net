"""
sample_adaptdiff.py
-------------------------------------
Sampling script for AdaptDiff-like conditional diffusion model.

Usage example:
    # label2image: generate synthetic target-like images
    python sample_adaptdiff.py --mode label2image --ckpt ../models/adaptdiff_label2image.pt \
        --input_dir ../preprocessed/pseudo_labels --output_dir ../preprocessed/generated_images

    # image2label: refine pseudo labels
    python sample_adaptdiff.py --mode image2label --ckpt ../models/adaptdiff_image2label.pt \
        --input_dir ../preprocessed/pseudo_labels --output_dir ../preprocessed/generated_labels
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
# Model definitions (same as training)
# ------------------------
class SimpleUNet(torch.nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base_ch=64):
        super().__init__()
        self.enc1 = torch.nn.Sequential(torch.nn.Conv2d(in_ch, base_ch, 3, padding=1), torch.nn.ReLU())
        self.enc2 = torch.nn.Sequential(torch.nn.Conv2d(base_ch, base_ch*2, 3, padding=1), torch.nn.ReLU())
        self.dec1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(base_ch*2, base_ch, 3, padding=1), torch.nn.ReLU())
        self.outc = torch.nn.Conv2d(base_ch, out_ch, 3, padding=1)

    def forward(self, x, t):
        # t is normalized scalar tensor shape (B,1,1,1)
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
    def p_sample(self, x_t, cond, t):
        """
        x_t: tensor shape [B, 1, H, W] (current noisy x)
        cond: tensor shape [B, 2, H, W] (conditioning channels)
        t: int scalar timestep
        Returns: next x_t (previous timestep) with shape [B,1,H,W]
        """
        # build model input matching training: concat x + cond => channels = 1 + 2 = 3
        x_concat = torch.cat([x_t, cond], dim=1)  # [B,3,H,W]
        t_norm = torch.tensor([t / float(self.timesteps)], device=x_t.device).repeat(x_t.shape[0])
        # model expects t shaped (B,) or (B,1,1,1) in our forward; we'll reshape in forward call
        t_norm = t_norm.view(-1, 1, 1, 1)
        pred_noise = self.model(x_concat, t_norm)  # pred_noise shape [B,1,H,W]

        # ddpm simple reverse step for the x channel only
        alpha_t = self.alphas_cumprod[t].to(x_t.device)
        sqrt_alpha = torch.sqrt(alpha_t)
        sqrt_one_minus = torch.sqrt(1 - alpha_t)
        # compute x_prev for the x channel
        x_prev = (x_t - sqrt_one_minus * pred_noise) / (sqrt_alpha + 1e-12)
        # return x_prev (keep cond unchanged externally)
        return x_prev

# ------------------------
# Dataset for slice-wise sampling (delayed load)
# ------------------------
class AdaptDiffDataset(torch.utils.data.Dataset):
    def __init__(self, folder, mode="label2image", target_size=(256,256)):
        self.files = sorted([os.path.join(folder,f) for f in os.listdir(folder) if f.endswith(".h5")])
        self.mode = mode
        self.target_size = target_size
        self.slices = []        # list of (fidx, z)
        self.slices_per_case = []  # list of n_slices per file (indexed by fidx)

        # build index
        for fidx, fpath in enumerate(self.files):
            with h5py.File(fpath, "r") as f:
                n_slices = f["image"].shape[2]
                self.slices_per_case.append(n_slices)
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
            # label might be missing for pure test volumes — caller expected pseudo_labels folder so label exists
            lbl = resize(f["label"][:,:,z], self.target_size, anti_aliasing=True).astype(np.float32) if "label" in f else np.zeros_like(img, dtype=np.float32)
            prob = resize(f["pseudo_prob"][:,:,z], self.target_size, anti_aliasing=True).astype(np.float32) if "pseudo_prob" in f else np.zeros_like(img, dtype=np.float32)

        img = torch.from_numpy(img).unsqueeze(0)   # [1,H,W]
        lbl = torch.from_numpy(lbl).unsqueeze(0)   # [1,H,W]
        prob = torch.from_numpy(prob).unsqueeze(0) # [1,H,W]

        # --- IMPORTANT: produce cond with exactly 2 channels ---
        if self.mode == "label2image":
            # cond: [lbl, zeros]  (2 channels)
            cond = torch.cat([lbl, torch.zeros_like(lbl)], dim=0)
        else:  # image2label
            # cond: [img, prob]  (2 channels)
            cond = torch.cat([img, prob], dim=0)

        # return cond, (img,lbl), fidx, z
        return cond, (img, lbl), int(fidx), int(z)

# ------------------------
# Sampling loop (grouped and write case-wise)
# ------------------------
@torch.no_grad()
def sample_adaptdiff(ckpt_path, input_dir, output_dir, mode="label2image",
                      num_steps=50, batch_size=8, target_size=(256,256), device_str=None):
    # device selection
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # accelerate conv when input sizes stable
    torch.backends.cudnn.benchmark = True

    os.makedirs(output_dir, exist_ok=True)

    # model channels: training used total input channels = 1 (x) + 2 (cond) = 3
    in_ch = 3
    out_ch = 1

    # build model and load checkpoint
    unet = SimpleUNet(in_ch=in_ch, out_ch=out_ch).to(device)
    diffusion = DiffusionModel(unet).to(device)

    print("[*] Loading checkpoint:", ckpt_path)
    state_dict = torch.load(ckpt_path, map_location=device)
    # state_dict might be full state_dict or raw model.state_dict(); attempt to load both
    try:
        diffusion.load_state_dict(state_dict)
    except RuntimeError:
        # maybe checkpoint wrapped in dict with key 'model_state' or similar
        if isinstance(state_dict, dict):
            # try common keys
            for key in ["model_state", "state_dict", "model_state_dict"]:
                if key in state_dict:
                    diffusion.load_state_dict(state_dict[key])
                    break
            else:
                # fallback: try to find nested dict that matches
                print("Warning: couldn't find nested 'model_state' key; trying to filter matching keys...")
                filtered = {k.replace("module.", ""): v for k, v in state_dict.items() if k.replace("module.", "") in diffusion.state_dict()}
                diffusion.load_state_dict(filtered)
        else:
            raise

    diffusion.eval()
    print("[*] Model loaded. Using device:", device)

    # dataset + loader
    dataset = AdaptDiffDataset(input_dir, mode=mode, target_size=target_size)
    # prepare number of cases
    n_cases = len(dataset.slices_per_case)
    n_slices_per_case = dataset.slices_per_case

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=4, pin_memory=True)

    # accumulate per-case slices: dict fidx -> list of (z, slice_array)
    case_outputs = {i: [] for i in range(n_cases)}

    print(f"[*] Sampling {len(dataset)} slices in {n_cases} cases (batch_size={batch_size}, steps={num_steps})")

    scaler = torch.cuda.amp.GradScaler(enabled=False)  # not used for @torch.no_grad(), keep for API parity
    # main loop
    for cond_batch, (img_batch, lbl_batch), fidx_batch, z_batch in tqdm(loader):
        # cond_batch: [B,2,H,W]; img_batch/lbl_batch: [B,1,H,W]
        cond_batch = cond_batch.to(device)          # [B,2,H,W]
        b = cond_batch.shape[0]
        H = cond_batch.shape[2]; W = cond_batch.shape[3]

        # x_t initial: random noise for the x channel only -> shape [B,1,H,W]
        x_t = torch.randn((b, 1, H, W), device=device)

        # iterative reverse process
        for t in reversed(range(num_steps)):
            # build input inside p_sample (we pass x_t and cond)
            # p_sample will concat internally and return next x_t (first channel)
            # Use autocast for speed (float16 inference)
            with torch.amp.autocast(device_type=str(device).split(':')[0] if device.type == 'cuda' else 'cpu'):
                x_t = diffusion.p_sample(x_t, cond_batch, t)

        out_batch = x_t.cpu().numpy()  # [B,1,H,W]
        # collect outputs per (fidx,z)
        for i in range(b):
            fidx = int(fidx_batch[i].item())
            z = int(z_batch[i].item())
            slice_arr = np.clip(out_batch[i, 0], 0.0, 1.0).astype(np.float32)  # [H,W]
            case_outputs[fidx].append((z, slice_arr))

    # write case-wise HDF5 files in order case000..caseNNN
    print("[*] Writing case-wise outputs ...")
    for case_id in range(n_cases):
        slices = case_outputs.get(case_id, [])
        n_slices_expected = n_slices_per_case[case_id]
        if len(slices) == 0:
            print(f"[WARN] case{case_id:03d} produced 0 slices, skipping.")
            continue
        # create empty volume and fill
        H, W = slices[0][1].shape
        vol = np.zeros((H, W, n_slices_expected), dtype=np.float32)
        for z, arr in slices:
            if 0 <= z < n_slices_expected:
                vol[:, :, z] = arr
            else:
                print(f"[WARN] case{case_id:03d} has slice index {z} out of bounds (0..{n_slices_expected-1})")
        case_name = f"case{case_id:03d}.h5"
        out_path = os.path.join(output_dir, case_name)
        if mode == "label2image":
            save_h5(out_path, {"image": vol})
        else:
            # binarize to hard labels (threshold 0.5)
            save_h5(out_path, {"label": (vol > 0.5).astype(np.uint8)})
        print(f"Saved {case_name} -> {out_path}  (shape {vol.shape})")

    print("[*] Sampling finished.")

# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fast sampling for AdaptDiff model")
    parser.add_argument("--ckpt", type=str, default=r"..\models\adaptdiff_image2label.pt")
    parser.add_argument("--input_dir", type=str, default=r"..\preprocessed\pseudo_labels")
    parser.add_argument("--output_dir", type=str, default=r"..\preprocessed\generated_labels")
    parser.add_argument("--mode", type=str, choices=["label2image","image2label"], default="image2label")
    parser.add_argument("--num_steps", type=int, default=50, help="Diffusion sampling steps (reduce to speed-up)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for slice-wise sampling")
    parser.add_argument("--target_size", type=int, nargs=2, default=(256,256), help="Resize spatially (H W)")
    parser.add_argument("--device", type=str, default=None, help="CUDA device string, e.g. 'cuda:0' or 'cpu'")
    args = parser.parse_args()

    sample_adaptdiff(args.ckpt, args.input_dir, args.output_dir, args.mode,
                     num_steps=args.num_steps, batch_size=args.batch_size,
                     target_size=tuple(args.target_size), device_str=args.device)
