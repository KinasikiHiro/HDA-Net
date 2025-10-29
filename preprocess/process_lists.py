import os
import nibabel as nib
import numpy as np
import h5py
from tqdm import tqdm
from skimage import exposure
import argparse

# =========================
# LiTS Preprocessing Script
# =========================
# This script prepares the LiTS dataset for domain adaptation and segmentation tasks.
# It loads .nii volumes, normalizes them, optionally resizes them,
# and saves them as HDF5 (.h5) files for efficient training.
#
# Expected directory structure:
# data/
# ├── train/
# │   ├── volume-0.nii
# │   ├── segmentation-0.nii
# │   └── ...
# └── test/
#     ├── test-volume-0.nii
#     └── ...
#
# Output:
# preprocessed/
# ├── train/
# │   ├── case_000.h5
# │   └── ...
# └── test/
#     ├── case_000.h5
#     └── ...
# =========================

def normalize_intensity(volume, clip_lower=-200, clip_upper=250):
    """
    Normalize the CT image intensity values to [0, 1].

    Args:
        volume (np.ndarray): Raw CT volume data.
        clip_lower (int): Lower bound for intensity clipping (default: -200 HU).
        clip_upper (int): Upper bound for intensity clipping (default: 250 HU).
    Returns:
        np.ndarray: Normalized volume in [0, 1].
    """
    volume = np.clip(volume, clip_lower, clip_upper)
    volume = (volume - clip_lower) / (clip_upper - clip_lower)
    return volume


def process_volume(volume_path, seg_path=None, save_path=None):
    """
    Load, normalize, and save a LiTS volume (and segmentation if available) as .h5.

    Args:
        volume_path (str): Path to the .nii volume file.
        seg_path (str or None): Path to the corresponding segmentation .nii file (if any).
        save_path (str): Path to save the processed .h5 file.
    """
    volume_nii = nib.load(volume_path)
    volume = volume_nii.get_fdata().astype(np.float32)

    volume = normalize_intensity(volume)

    # Optional: histogram equalization (slightly improves low-dose visibility)
    volume = exposure.equalize_adapthist(volume)

    if seg_path:
        seg_nii = nib.load(seg_path)
        seg = seg_nii.get_fdata().astype(np.uint8)
    else:
        seg = None

    with h5py.File(save_path, 'w') as f:
        f.create_dataset('image', data=volume, compression="gzip")
        if seg is not None:
            f.create_dataset('label', data=seg, compression="gzip")

    print(f"Saved preprocessed file: {save_path}")


def preprocess_lits_dataset(root_dir='../data', output_dir='../preprocessed',
                            start_train_idx=0, start_test_idx=0):
    """
    Preprocess the LiTS dataset and convert all volumes to HDF5 format.

    Args:
        root_dir (str): Root path of the LiTS dataset.
        output_dir (str): Path to save the processed .h5 files.
    """
    train_input_dir = os.path.join(root_dir, 'train')
    test_input_dir = os.path.join(root_dir, 'test')
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

    # ---- Process training set (with segmentation labels) ----
    print("Processing training data...")
    for i in tqdm(range(131)):
        vol_path = os.path.join(train_input_dir, f'volume-{i}.nii')
        seg_path = os.path.join(train_input_dir, f'segmentation-{i}.nii')
        save_path = os.path.join(output_dir, 'train', f'case_{i:03d}.h5')

        # 跳过已存在文件
        if os.path.exists(save_path):
            print(f"Skipping existing: {save_path}")
            continue

        process_volume(vol_path, seg_path, save_path)

    # ---- Process test set (no segmentation labels) ----
    print("Processing test data...")
    for i in tqdm(range(70)):
        vol_path = os.path.join(test_input_dir, f'test-volume-{i}.nii')
        save_path = os.path.join(output_dir, 'test', f'case_{i:03d}.h5')

        if os.path.exists(save_path):
            print(f"Skipping existing: {save_path}")
            continue

        process_volume(vol_path, seg_path=None, save_path=save_path)

    print("Preprocessing complete! All files saved to:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiTS Preprocessing (Restart-safe)")
    parser.add_argument("--root_dir", type=str, default="../data", help="LiTS raw data directory")
    parser.add_argument("--output_dir", type=str, default="../preprocessed", help="Output directory")
    parser.add_argument("--start_train_idx", type=int, default=0, help="Start index for training set")
    parser.add_argument("--start_test_idx", type=int, default=0, help="Start index for test set")

    args = parser.parse_args()

    preprocess_lits_dataset(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        start_train_idx=args.start_train_idx,
        start_test_idx=args.start_test_idx
    )
