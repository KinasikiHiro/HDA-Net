"""
da_analysis.py
Domain-gap analysis utilities.
Assumes preprocessed files in HDA-Net/preprocessed/train/*.h5 and HDA-Net/preprocessed/test/*.h5
or you can adapt to read .nii via nibabel.

Dependencies: numpy, h5py, matplotlib, scipy, skimage
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter
from skimage.feature import hog

# ---------------------------
# 1) IO helpers
# ---------------------------
def load_h5_image(path):
    """
    Load a preprocessed .h5 file and return numpy arrays.

    Args:
        path (str): path to .h5 file
    Returns:
        image (np.ndarray): 3D image float32, range [0,1]
        label (np.ndarray or None): 3D label array (if present)
    """
    with h5py.File(path, 'r') as f:
        img = f['image'][()]
        lbl = f['label'][()] if 'label' in f else None
    return img.astype(np.float32), (lbl.astype(np.uint8) if lbl is not None else None)


# ---------------------------
# 2) Basic statistics & histograms
# ---------------------------
def volume_intensity_stats(image):
    """
    Compute basic intensity statistics for a 3D volume.

    Returns:
        dict: mean, std, median, p10, p90, skewness (approx), kurtosis (approx)
    """
    arr = image.flatten()
    arr = arr[~np.isnan(arr)]
    mean = arr.mean()
    std = arr.std()
    median = np.median(arr)
    p10 = np.percentile(arr, 10)
    p90 = np.percentile(arr, 90)
    # simple skew/kurt via scipy could be used; here use moments
    from scipy.stats import skew, kurtosis
    sk = skew(arr)
    kur = kurtosis(arr)
    return dict(mean=mean, std=std, median=median, p10=p10, p90=p90, skew=sk, kurtosis=kur)


def plot_histograms(image_list, labels, out_png=None, nbins=256):
    """
    Plot overlaid histograms / CDFs for a list of images.

    Args:
        image_list (list[np.ndarray]): list of 3D volumes
        labels (list[str]): labels for legend
    """
    plt.figure(figsize=(10,4))
    for img, lab in zip(image_list, labels):
        flat = img.flatten()
        plt.hist(flat, bins=nbins, density=True, alpha=0.5, label=lab)
    plt.legend()
    plt.title("Intensity histograms")
    if out_png:
        plt.savefig(out_png, dpi=200)
    else:
        plt.show()


def kl_divergence_between_histograms(img_src, img_tgt, nbins=256, eps=1e-10):
    """
    Compute KL divergence between intensity distributions of two volumes.

    Returns:
        float: KL(P||Q) where P=src hist, Q=tgt hist
    """
    p_hist, _ = np.histogram(img_src.flatten(), bins=nbins, density=True)
    q_hist, _ = np.histogram(img_tgt.flatten(), bins=nbins, density=True)
    p = p_hist + eps
    q = q_hist + eps
    return entropy(p, q)  # KL(P||Q)


# ---------------------------
# 3) Noise / SNR estimation
# ---------------------------
def estimate_noise_std_in_background(image, bg_mask=None):
    """
    Estimate noise std. If bg_mask provided, compute std over that mask;
    otherwise use small patches with low variance as proxy for background.

    Args:
        image (np.ndarray): 3D volume
        bg_mask (np.ndarray or None): binary mask of background/air region
    Returns:
        float: estimated noise std (in normalized units)
    """
    if bg_mask is not None:
        vals = image[bg_mask > 0]
        return float(np.std(vals))
    # else heuristic: find patches with smallest local std
    from skimage.util import view_as_windows
    # sample some slices
    slice_idxs = np.linspace(0, image.shape[2]-1, min(20, image.shape[2])).astype(int)
    stds = []
    for s in slice_idxs:
        sl = image[:,:,s]
        # compute gaussian blurred version and subtract to estimate high freq
        lf = gaussian_filter(sl, sigma=3)
        hf = sl - lf
        stds.append(np.std(hf))
    return float(np.median(stds))


def compute_snr(image, signal_mask):
    """
    Compute SNR = mean(signal)/std(background)
    Requires a signal_mask (e.g., liver ROI) and background mask.
    """
    signal = image[signal_mask > 0]
    # choose background as voxels far from liver mask (negation)
    bg = image[signal_mask == 0]
    return float(signal.mean() / (bg.std() + 1e-8))


# ---------------------------
# 4) Feature-level distance (slice-based using simple CNN features or precomputed encoder)
# ---------------------------
def slice_feature_distance(img_src, img_tgt, slice_axis=2, n_slices=20):

    min_slices = min(img_src.shape[slice_axis], img_tgt.shape[slice_axis])
    zs = np.linspace(0, min_slices-1, n_slices).astype(int)
    dists = []
    for z in zs:
        if slice_axis == 2:
            s1 = img_src[:,:,z]
            s2 = img_tgt[:,:,z]
        elif slice_axis == 0:
            s1 = img_src[z,:,:]
            s2 = img_tgt[z,:,:]
        else:
            s1 = img_src[:,z,:]; s2 = img_tgt[:,z,:]
        # resize to manageable size
        from skimage.transform import resize
        s1r = resize(s1, (256,256), anti_aliasing=True)
        s2r = resize(s2, (256,256), anti_aliasing=True)
        f1 = hog(s1r, pixels_per_cell=(16,16), cells_per_block=(2,2), feature_vector=True)
        f2 = hog(s2r, pixels_per_cell=(16,16), cells_per_block=(2,2), feature_vector=True)
        dists.append(np.linalg.norm(f1 - f2))
    return float(np.mean(dists))


# ---------------------------
# 5) Downstream test: compute source-trained segmentation performance on target (requires model)
# ---------------------------
# This must be done in training pipeline: train on LiTS, save model, run inference on target and compute Dice.
# Here we only provide a placeholder utility to compute Dice if predictions exist.
def dice_coefficient(pred, gt, eps=1e-6):
    pred_bin = (pred > 0.5).astype(np.int)
    gt_bin = (gt > 0).astype(np.int)
    inter = (pred_bin & gt_bin).sum()
    return 2.0 * inter / (pred_bin.sum() + gt_bin.sum() + eps)


# ---------------------------
# Example driving function
# ---------------------------
def analyze_pair(h5_src_path, h5_tgt_path, out_dir=None):
    """
    Analyze one source-target volume pair (not necessarily same patient).
    Produces histograms, KL, noise estimates, feature-distance, saves plots.
    """
    img_s, _ = load_h5_image(h5_src_path)
    img_t, _ = load_h5_image(h5_tgt_path)

    stats_s = volume_intensity_stats(img_s)
    stats_t = volume_intensity_stats(img_t)
    kl = kl_divergence_between_histograms(img_s, img_t)
    noise_s = estimate_noise_std_in_background(img_s)
    noise_t = estimate_noise_std_in_background(img_t)
    feat_dist = slice_feature_distance(img_s, img_t)

    print("Source stats:", stats_s)
    print("Target stats:", stats_t)
    print("KL divergence (intensity):", kl)
    print("Noise std: src {:.4f}, tgt {:.4f}".format(noise_s, noise_t))
    print("Feature distance (HOG proxy):", feat_dist)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        plot_histograms([img_s, img_t], ['src', 'tgt'], out_png=os.path.join(out_dir, "hist.png"))
        # save some slice comparison images
        z = img_s.shape[2] // 2
        plt.figure(figsize=(10,4))
        plt.subplot(1,3,1); plt.imshow(img_s[:,:,z], cmap='gray'); plt.title('src mid-slice')
        plt.subplot(1,3,2); plt.imshow(img_t[:,:,z], cmap='gray'); plt.title('tgt mid-slice')
        plt.subplot(1,3,3); plt.imshow(np.abs(img_s[:,:,z]-img_t[:,:,z]), cmap='inferno'); plt.title('abs diff')
        plt.savefig(os.path.join(out_dir, "slice_comp.png"))

    return dict(stats_src=stats_s, stats_t=stats_t, kl=kl, noise_src=noise_s, noise_t=noise_t, feat_dist=feat_dist)
