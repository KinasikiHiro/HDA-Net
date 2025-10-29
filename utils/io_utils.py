"""
io.py - I/O utilities for HDA-Net project (PyTorch)

Contains:
 - HDF5 dataset loader (H5Dataset)
 - helper functions: load_h5, save_h5, collate_fn, basic transforms
 - note: expects preprocessed .h5 files produced by process_lits.py,
   where each .h5 contains dataset 'image' (z,y,x) and optional 'label' (z,y,x).
"""

import os
from typing import Optional, Tuple, List
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import random

# ------------------------
# Basic IO functions
# ------------------------
def load_h5(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load an .h5 file saved by our preprocessor.
    Args:
        path: path to .h5 file
    Returns:
        image: numpy array (z, y, x) dtype float32 (values in [0,1])
        label: numpy array (z, y, x) dtype int16 or None (if not present)
    """
    with h5py.File(path, 'r') as f:
        image = f['image'][()]
        label = f['label'][()] if 'label' in f else None
    return image.astype(np.float32), (label.astype(np.int16) if label is not None else None)

def save_h5(path: str, image: np.ndarray, label: Optional[np.ndarray] = None):
    """
    Save image and optional label into an h5 file.
    Args:
        path: output path
        image: (z,y,x) float32
        label: (z,y,x) int or None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset('image', data=image, compression='gzip')
        if label is not None:
            f.create_dataset('label', data=label, compression='gzip')

# ------------------------
# Dataset class
# ------------------------
class H5Dataset(Dataset):
    """
    PyTorch Dataset for .h5 volumes.
    Each sample returns:
      - image: torch.FloatTensor (1, D, H, W) where D = num_slices (z)
      - label: torch.LongTensor (D, H, W) if available, else None

    Options:
      - crop_size: tuple (dz, dh, dw) for random crop (if provided)
      - augment: bool, if True apply simple flips
    """
    def __init__(self, folder: str, files: Optional[List[str]] = None, crop_size: Optional[tuple] = None, augment: bool = False):
        super().__init__()
        self.folder = folder
        if files is None:
            self.files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.h5')])
        else:
            self.files = files
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def random_crop(self, image: np.ndarray, label: Optional[np.ndarray]):
        """Randomly crop image/label to crop_size. Image shape: (z,y,x)."""
        dz, dy, dx = self.crop_size
        z, y, x = image.shape
        if z <= dz:
            z0 = 0
        else:
            z0 = random.randint(0, z - dz)
        if y <= dy:
            y0 = 0
        else:
            y0 = random.randint(0, y - dy)
        if x <= dx:
            x0 = 0
        else:
            x0 = random.randint(0, x - dx)
        img_c = image[z0:z0+dz, y0:y0+dy, x0:x0+dx]
        lbl_c = label[z0:z0+dz, y0:y0+dy, x0:x0+dx] if label is not None else None
        return img_c, lbl_c

    def __getitem__(self, idx):
        path = self.files[idx]
        image, label = load_h5(path)
        # optional simple augmentations
        if self.augment:
            if random.random() < 0.5:
                image = np.flip(image, axis=1).copy()
                if label is not None: label = np.flip(label, axis=1).copy()
            if random.random() < 0.5:
                image = np.flip(image, axis=2).copy()
                if label is not None: label = np.flip(label, axis=2).copy()

        if self.crop_size is not None:
            image, label = self.random_crop(image, label)

        # To torch: shape (C=1, D, H, W)
        img_t = torch.from_numpy(image).unsqueeze(0).float()
        lbl_t = torch.from_numpy(label).long() if label is not None else torch.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=torch.long)
        return img_t, lbl_t

# ------------------------
# Collate fn for dataloader (simple)
# ------------------------
def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.stack([b[1] for b in batch], dim=0)
    return imgs, labels

# ------------------------
# small utils
# ------------------------
def list_h5_files(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.h5')])
