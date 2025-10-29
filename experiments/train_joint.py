"""
train_joint.py
-------------------------------------
Joint training: Segmentation (HybridHDenseUNet) + AdaptDiff (image2label mode).

Workflow (high level):
  1) If generated_target_like/ not exist, run diffusion sampler (batch slice-wise) to create
     generated volumes from pseudo_labels (image+prob -> refined label or image).
  2) Load source dataset (preprocessed/train) as H5Dataset (same as your train_h_denseunet).
  3) Load generated dataset as simple H5 loader.
  4) Train segmentation network with mixed batches (source supervised + generated pseudo-supervised).
  5) Optionally fine-tune diffusion simultaneously (use --finetune_diff).

Notes:
  - Designed to be conservative — if your local class/IO names differ slightly, adjust imports.
  - Keep image2label mode as default.
"""

import os
import sys
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import h5py

# path setup (same style as your other scripts)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
UTILS_DIR = os.path.join(ROOT_DIR, "utils")
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

# try import your H5Dataset and collate (from train_h_denseunet / io_utils)
try:
    from io_utils import H5Dataset, collate_fn  # your utils/io_utils.py
except Exception:
    # fallback: simple volume dataset (expects full h5 volumes in folder)
    H5Dataset = None
    collate_fn = None

# import your HybridHDenseUNet model (from experiments/train_h_denseunet.py)
sys.path.append(os.path.join(ROOT_DIR, "experiments"))
from train_h_denseunet import HybridHDenseUNet

# --- small diffusion model definitions (copy of train_adaptdiff style) ---
class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base_ch=64):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, base_ch * 2, 3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(base_ch * 2, base_ch, 3, padding=1), nn.ReLU())
        self.outc = nn.Conv2d(base_ch, out_ch, 3, padding=1)

    def forward(self, x, t):
        # t is normalized scalar or tensor of shape (B,)
        t_emb = t.view(-1,1,1,1)
        x = x + t_emb
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.dec1(x)
        x = self.outc(x)
        return x

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
        sqrt_acp = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
        sqrt_omp = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]
        return sqrt_acp * x0 + sqrt_omp * noise

    @torch.no_grad()
    def p_sample(self, x_t, cond, t):
        # cond: (B, C_cond, H, W). x_t: (B, out_ch, H, W)
        t_norm = torch.tensor([t / float(self.timesteps)], device=x_t.device, dtype=x_t.dtype)
        # model expects concatenated channels [x_t, cond]
        inp = torch.cat([x_t, cond], dim=1)
        pred_noise = self.model(inp, t_norm)
        alpha_t = self.alphas_cumprod[t]
        x_prev = (x_t - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()
        return x_prev

    def denoise_steps(self, cond, num_steps=25, device='cuda'):
        # utility to sample from pure noise -> x0 (batch)
        B = cond.shape[0]
        out_ch = 1
        x_t = torch.randn((B, out_ch, cond.shape[2], cond.shape[3]), device=device)
        for t in reversed(range(num_steps)):
            x_t = self.p_sample(x_t, cond, t)
        return x_t

# -------------------------
# small helper IO for generated dataset
# -------------------------
def save_h5(path, data_dict):
    with h5py.File(path, "w") as f:
        for k, v in data_dict.items():
            f.create_dataset(k, data=v, compression="gzip")

def load_h5_simple(path):
    with h5py.File(path, "r") as f:
        return {k: f[k][()] for k in f.keys()}

# -------------------------
# Generated volumes dataset (expects generated volumes saved as .h5 with keys 'label' and/or 'image')
# -------------------------
class GeneratedVolumeDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".h5")])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        data = load_h5_simple(self.files[idx])
        # anticipate keys: 'image' and/or 'label'
        image = data.get('image')  # may be None
        label = data.get('label')
        # ensure shape is (C,D,H,W) for segmentation loader compatibility
        # we assume saved arrays are (H,W,D) or (D,H,W); handle (H,W,D) by checking dims
        # Most of your code used (H,W,D) stack when saving slices; we keep that convention
        if label is not None:
            # label: H x W x D  -> convert to (1, D, H, W)
            if label.ndim == 3:
                H,W,D = label.shape
                label = label.transpose(2,0,1)  # -> (D,H,W)
                label = label[None].astype(np.int16)  # (1,D,H,W)
            elif label.ndim == 4:
                # assume (C,D,H,W)
                pass
        if image is not None:
            if image.ndim == 3:
                image = image.transpose(2,0,1)[None].astype(np.float32)  # (1,D,H,W)
        # return torch tensors
        img_t = torch.from_numpy(image).float() if image is not None else None
        lbl_t = torch.from_numpy(label).long() if label is not None else None
        return img_t, lbl_t

# -------------------------
# Sampling utility (re-uses DiffusionModel) to create generated volumes (per-case)
# -------------------------
def sample_and_save_volumes(diffusion, ckpt_path, input_dir, output_dir,
                            mode="image2label", num_steps=25, batch_size=8, device='cuda'):
    """
    Slice-wise sampling and case-wise assembly.
    - input_dir: preprocessed/pseudo_labels (expects per-case .h5 with fields image,label,pseudo_prob)
    - output_dir: directory to write caseXXX.h5 containing 'label' (and 'image' optional)
    """
    os.makedirs(output_dir, exist_ok=True)
    # load diffusion state (already loaded outside; ckpt_path provided only for logging)
    print(f"[Sampler] sampling from diffusion checkpoint: {ckpt_path}")
    # Build dataset same as in your sample_adaptdiff (slice-wise)
    files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".h5")])
    # build list of (file_idx, z)
    slices = []
    vol_slices_count = []
    for fidx, fpath in enumerate(files):
        with h5py.File(fpath, "r") as ff:
            n = ff['image'].shape[2]
            vol_slices_count.append(n)
            for z in range(n):
                slices.append((fidx, z))
    # create a loader that yields batches of (fidx, z) indices
    from skimage.transform import resize
    BATCH = batch_size
    idxs = list(range(len(slices)))
    # We'll accumulate per-case slices to lists then stack per case
    case_outputs = {i: [] for i in range(len(files))}
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    diffusion.eval()
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(idxs), BATCH), desc="[Sampler] batches"):
            batch_idxs = idxs[batch_start:batch_start+BATCH]
            cond_list = []
            fidx_list = []
            for i in batch_idxs:
                fidx, z = slices[i]
                fidx_list.append((fidx, z))
                with h5py.File(files[fidx], "r") as f:
                    # read image (H,W) and pseudo_prob (H,W)
                    img = f['image'][:,:,z].astype(np.float32)
                    prob = f['pseudo_prob'][:,:,z].astype(np.float32) if 'pseudo_prob' in f else np.zeros_like(img, dtype=np.float32)
                    # resize to 256x256 for diffusion (match your training)
                    img_rs = resize(img, (256,256), anti_aliasing=True).astype(np.float32)
                    prob_rs = resize(prob, (256,256), anti_aliasing=True).astype(np.float32)
                    # cond for image2label: [img, prob] channels
                    cond = np.stack([img_rs, prob_rs], axis=0)  # (2,H,W)
                    cond_list.append(cond)
            cond_batch = torch.from_numpy(np.stack(cond_list, axis=0)).to(device)  # (B,2,H,W)
            # sample (denoise) to get labels or logits (out channels = 1)
            out_batch = diffusion.denoise_steps(cond_batch, num_steps=num_steps, device=device)  # (B,1,H,W)
            out_batch = out_batch.cpu().numpy()  # (B,1,H,W)
            # postprocess and store per fidx/z
            for ii, (fidx, z) in enumerate(fidx_list):
                out_slice = out_batch[ii,0]
                # Resize back to original slice size (use nearest or bilinear)
                with h5py.File(files[fidx], "r") as f:
                    H0, W0 = f['image'].shape[0], f['image'].shape[1]
                out_resized = resize(out_slice, (H0, W0), anti_aliasing=True).astype(np.float32)
                case_outputs[fidx].append((z, out_resized))
    # assemble per-case and save
    for fidx, items in case_outputs.items():
        if not items:
            continue
        # items is list of (z, slice_array) but may be unordered; sort by z
        items_sorted = sorted(items, key=lambda x: x[0])
        slices_only = [it[1] for it in items_sorted]
        vol = np.stack(slices_only, axis=2)  # (H,W,D)
        case_name = f"case{fidx:03d}.h5"
        out_path = os.path.join(output_dir, case_name)
        # save label volume (thresholded) as uint8
        label_vol = (vol > 0.5).astype(np.uint8)
        save_h5(out_path, {'label': label_vol})
        print(f"[Sampler] saved {out_path} shape {label_vol.shape}")

# -------------------------
# Joint training entrypoint
# -------------------------
def train_joint(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ----------------------
    # 1) Prepare generated data (if not exist)
    # ----------------------
    generated_dir = os.path.abspath(args.generated_dir)
    if not os.path.exists(generated_dir) or len(os.listdir(generated_dir)) == 0:
        print("[Joint] generated_dir empty — running sampler to create generated volumes ...")
        # load diffusion model
        cond_channels = 2  # image + prob
        model_in_ch = 1 + cond_channels  # xt channel + cond channels
        unet = SimpleUNet(in_ch=model_in_ch, out_ch=1)
        diffusion = DiffusionModel(unet).to(device)
        # load checkpoint
        if os.path.exists(args.diff_ckpt):
            state = torch.load(args.diff_ckpt, map_location=device)
            # try load directly; checkpoint may be state_dict or saved model_state
            if isinstance(state, dict) and ('model_state' in state or 'state_dict' in state):
                to_load = state.get('model_state', state.get('state_dict', state))
            else:
                to_load = state
            diffusion.load_state_dict(to_load)
            print("[Joint] loaded diffusion checkpoint for sampling.")
        else:
            raise FileNotFoundError(f"Diffusion ckpt not found: {args.diff_ckpt}")
        # run sampling (writes caseXXX.h5 under generated_dir)
        sample_and_save_volumes(diffusion, args.diff_ckpt, args.pseudo_dir, generated_dir,
                                mode="image2label", num_steps=args.num_steps,
                                batch_size=args.sample_batch, device=device)

    # ----------------------
    # 2) Prepare datasets & models
    # ----------------------
    # Source dataset (H5Dataset) - expects labelled volumes
    src_dir = os.path.join(ROOT_DIR, "preprocessed", "train")
    if H5Dataset is not None:
        src_ds = H5Dataset(src_dir, crop_size=(args.crop_d, args.crop_h, args.crop_w), augment=True)
        src_loader = DataLoader(src_ds, batch_size=args.src_batch, shuffle=True,
                                num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    else:
        raise RuntimeError("H5Dataset (io_utils) not found — please ensure utils/io_utils.py is importable")

    # Generated dataset
    gen_ds = GeneratedVolumeDataset(generated_dir)
    gen_loader = DataLoader(gen_ds, batch_size=args.gen_batch, shuffle=True, num_workers=max(0,args.num_workers//2), pin_memory=True)

    # Models: segmentation & diffusion
    seg_net = HybridHDenseUNet(encoder_out_ch=args.enc_ch, n_classes=args.num_classes, pretrained_encoder=False).to(device)
    seg_optimizer = optim.Adam(seg_net.parameters(), lr=args.lr_seg)

    # Optional: fine-tune diffusion
    if args.finetune_diff:
        cond_channels = 2
        model_in_ch = 1 + cond_channels
        unet2 = SimpleUNet(in_ch=model_in_ch, out_ch=1)
        diffusion = DiffusionModel(unet2).to(device)
        if os.path.exists(args.diff_ckpt):
            state = torch.load(args.diff_ckpt, map_location=device)
            if isinstance(state, dict) and ('model_state' in state or 'state_dict' in state):
                to_load = state.get('model_state', state.get('state_dict', state))
            else:
                to_load = state
            diffusion.load_state_dict(to_load, strict=False)
            print("[Joint] loaded diffusion (finetune mode).")
        diff_optimizer = optim.Adam(diffusion.parameters(), lr=args.lr_diff)
    else:
        diffusion = None
        diff_optimizer = None

    # Losses
    ce_loss = nn.CrossEntropyLoss()
    def dice_loss(pred, target, eps=1e-6):
        probs = F.softmax(pred, dim=1)
        B,C,D,H,W = probs.shape
        target_onehot = torch.zeros((B,C,D,H,W), device=pred.device)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)
        inter = (probs * target_onehot).sum(dim=(2,3,4))
        card = probs.sum(dim=(2,3,4)) + target_onehot.sum(dim=(2,3,4))
        dice = (2*inter + eps)/(card + eps)
        return 1 - dice.mean()

    # ----------------------
    # 3) Joint training loop (epoch-wise)
    # ----------------------
    num_epochs = args.epochs
    best_val = 1e9
    for epoch in range(num_epochs):
        seg_net.train()
        if diffusion is not None:
            diffusion.train()
        epoch_loss = 0.0
        n_steps = 0

        # zip source and generated loaders (cycling generated if shorter)
        gen_iter = iter(gen_loader)
        for src_batch in tqdm(src_loader, desc=f"Joint Epoch {epoch+1}/{num_epochs}"):
            imgs_src, labels_src = src_batch  # H5Dataset collate ensures proper shapes
            imgs_src = imgs_src.to(device); labels_src = labels_src.to(device)

            # get a generated batch (may need to restart iterator)
            try:
                imgs_gen, labels_gen = next(gen_iter)
            except StopIteration:
                gen_iter = iter(gen_loader)
                imgs_gen, labels_gen = next(gen_iter)
            # imgs_gen: (B,1,D,H,W) or None; labels_gen: (B,1,D,H,W)
            imgs_gen = imgs_gen.to(device) if imgs_gen is not None else None
            labels_gen = labels_gen.to(device)

            # --- forward seg on source ---
            seg_optimizer.zero_grad()
            outputs_src = seg_net(imgs_src)  # (B, C, D, H, W)
            loss_ce_src = ce_loss(outputs_src, labels_src)
            loss_dice_src = dice_loss(outputs_src, labels_src)
            loss_src = args.alpha * loss_ce_src + (1-args.alpha) * loss_dice_src

            # --- forward seg on generated (pseudo-supervised) ---
            outputs_gen = seg_net(imgs_gen)
            loss_ce_gen = ce_loss(outputs_gen, labels_gen.squeeze(1))
            loss_dice_gen = dice_loss(outputs_gen, labels_gen.squeeze(1))
            loss_gen = args.alpha * loss_ce_gen + (1-args.alpha) * loss_dice_gen

            total_loss = loss_src + args.lambda_gen * loss_gen

            # --- optionally fine-tune diffusion with diffusion self-supervision on generated batch ---
            if diffusion is not None and args.finetune_diff:
                # build cond for diffusion: image slices + pseudo_prob; here we approximate by slicing middle depth
                # We'll extract center-slices from imgs_gen and labels_gen to compute a quick diffusion loss
                B, _, D, H, W = imgs_gen.shape
                center = D//2
                # cond: image + prob (use label probabilities by softening one-hot)
                img_slice = imgs_gen[:,:,center,:,:].squeeze(1)  # (B,H,W)
                # build pseudo_prob as one-hot smoothed from labels_gen center slice
                label_slice = labels_gen[:,:,center,:,:].squeeze(1).float()
                # pseudo_prob: use label_slice as proxy probability (since we don't have softmax here)
                pseudo_prob = label_slice  # (B,H,W)
                cond_np = torch.stack([img_slice, pseudo_prob], dim=1)  # (B,2,H,W)
                # create noisy x0 and compute noise prediction target
                # create x0 = label_slice (as float) and t random
                x0 = label_slice.unsqueeze(1)  # (B,1,H,W)
                t_rand = torch.randint(0, diffusion.timesteps, (x0.shape[0],), device=device)
                noise = torch.randn_like(x0)
                xt = diffusion.q_sample(x0, t_rand, noise)
                inp = torch.cat([xt, cond_np], dim=1)
                # predict noise
                pred_noise = diffusion.model(inp, (t_rand.float()/diffusion.timesteps))
                loss_diff = F.mse_loss(pred_noise, noise)
                total_loss = total_loss + args.lambda_diff * loss_diff

            # backward & step
            total_loss.backward()
            seg_optimizer.step()
            if diffusion is not None and args.finetune_diff:
                diff_optimizer.step()

            epoch_loss += float(total_loss.item())
            n_steps += 1

        avg_loss = epoch_loss / max(1, n_steps)
        print(f"[Joint] Epoch {epoch+1} Avg Loss: {avg_loss:.5f}")

        # save segmentation checkpoint
        ckpt_dir = os.path.join(ROOT_DIR, "models", "joint")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({'epoch': epoch, 'seg_state': seg_net.state_dict()}, os.path.join(ckpt_dir, f"joint_seg_epoch{epoch+1:03d}.pt"))
        if diffusion is not None and args.finetune_diff:
            torch.save({'epoch': epoch, 'diff_state': diffusion.state_dict()}, os.path.join(ckpt_dir, f"joint_diff_epoch{epoch+1:03d}.pt"))

    print("[Joint] Training finished.")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Joint training: SegNet + AdaptDiff (image2label)")
    parser.add_argument("--diff_ckpt", type=str, default=r"..\models\adaptdiff_image2label.pt")
    parser.add_argument("--pseudo_dir", type=str, default=r"..\preprocessed\pseudo_labels")
    parser.add_argument("--generated_dir", type=str, default=r"..\preprocessed\generated_target_like")
    parser.add_argument("--num_steps", type=int, default=25, help="sampling steps for generating volumes (smaller -> faster)")
    parser.add_argument("--sample_batch", type=int, default=8, help="sampler batch size")
    parser.add_argument("--src_batch", type=int, default=1)
    parser.add_argument("--gen_batch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr_seg", type=float, default=1e-4)
    parser.add_argument("--lr_diff", type=float, default=5e-5)
    parser.add_argument("--alpha", type=float, default=0.5, help="CE vs Dice weight for segmentation")
    parser.add_argument("--lambda_gen", type=float, default=1.0, help="Weight for generated-data seg loss")
    parser.add_argument("--finetune_diff", action="store_true", help="Also finetune diffusion during joint training")
    parser.add_argument("--lambda_diff", type=float, default=0.1, help="Weight for diffusion self-supervision loss")
    parser.add_argument("--enc_ch", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    train_joint(args)
