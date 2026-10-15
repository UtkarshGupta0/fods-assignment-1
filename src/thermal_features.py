import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.measure import label as cc_label

from common import (cached_features, describe, log_result, make_split,
                    plot_embedding_2d, save_fig)
from rgb_images import cnn_embeddings, color_histogram_features, hog_features

N_PAIRS = 400
SIZE = (160, 128)


def _to_arrays(rgb_imgs, ir_imgs):
    rgb = np.stack([np.array(i.convert("RGB").resize(SIZE)) for i in rgb_imgs])
    ir = np.stack([np.array(i.convert("L").resize(SIZE)) for i in ir_imgs])
    return rgb, ir


def load_from_folders(rgb_dir, ir_dir, n=N_PAIRS):
    from PIL import Image
    ext = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp")
    a = sorted(sum([glob.glob(os.path.join(rgb_dir, e)) for e in ext], []))[:n]
    b = sorted(sum([glob.glob(os.path.join(ir_dir, e)) for e in ext], []))[:n]
    k = min(len(a), len(b))
    if k == 0:
        raise SystemExit("No images found in the given folders.")
    return _to_arrays([Image.open(p) for p in a[:k]], [Image.open(p) for p in b[:k]])


def load_flir_paired(n=N_PAIRS):
    from datasets import Image as HFImage, load_dataset
    ds = load_dataset("newguyme/flir_paired", split="test")
    img_cols = [k for k, v in ds.features.items() if isinstance(v, HFImage)]
    print("  image columns:", img_cols)
    low = {c: c.lower() for c in img_cols}
    t_col = next((c for c in img_cols if any(s in low[c] for s in ("thermal", "infrared", "ir"))), None)
    r_col = next((c for c in img_cols if any(s in low[c] for s in ("rgb", "visible", "vis", "color"))), None)
    if t_col is None or r_col is None or t_col == r_col:
        raise SystemExit(f"Could not tell which column is RGB / thermal from {img_cols}; "
                         f"edit r_col / t_col in load_flir_paired().")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    print(f"  using RGB column '{r_col}' and thermal column '{t_col}', {len(ds)} pairs")
    return _to_arrays(ds[r_col], ds[t_col])


def thermal_intensity_hist(ir, bins=32):
    h = np.stack([np.histogram(i, bins=bins, range=(0, 256))[0] for i in ir]).astype(np.float32)
    return h / h.sum(1, keepdims=True)


def hot_blob_features(ir):
    out = []
    for im in ir:
        t = threshold_otsu(im)
        hot = im > t
        lab, n = cc_label(hot, return_num=True)
        areas = np.bincount(lab.ravel())[1:] if n else np.array([0])
        hv = im[hot] if hot.any() else np.array([0])
        out.append([hot.mean(), n, areas.max() / im.size, hv.mean() / 255, hv.std() / 255, t / 255])
    return np.array(out, dtype=np.float32)


def to3(gray):
    return np.repeat(gray[..., None], 3, axis=-1)


def main():
    print("== THERMAL vs RGB (paired images) ==")
    if "--folders" in sys.argv:
        i = sys.argv.index("--folders")
        rgb, ir = load_from_folders(sys.argv[i + 1], sys.argv[i + 2])
    else:
        rgb, ir = load_flir_paired()
    describe("RGB (N,H,W,3)", rgb);
    describe("thermal (N,H,W)", ir)

    bright = rgb.mean(axis=(1, 2, 3))
    y = (bright > np.median(bright)).astype(int)
    names = ["dark scene", "bright scene"]
    split = make_split(y)

    fig, ax = plt.subplots(2, 4, figsize=(12, 5))
    order = np.argsort(bright)
    for k, j in enumerate([order[0], order[1], order[-2], order[-1]]):
        ax[0, k].imshow(rgb[j]);
        ax[0, k].set_title(f"RGB (brightness {bright[j]:.0f})", fontsize=8)
        ax[1, k].imshow(ir[j], cmap="inferno");
        ax[1, k].set_title("thermal", fontsize=8)
        ax[0, k].axis("off");
        ax[1, k].axis("off")
    plt.tight_layout();
    save_fig("thermal_vs_rgb_pairs.png")

    feats = {
        "RGB colour histogram": lambda: color_histogram_features(rgb),
        "RGB HOG": lambda: hog_features(rgb),
        "RGB ResNet50": lambda: cnn_embeddings(rgb),
        "Thermal intensity histogram": lambda: thermal_intensity_hist(ir),
        "Thermal hot-blob stats": lambda: hot_blob_features(ir),
        "Thermal HOG": lambda: hog_features(to3(ir)),
        "Thermal ResNet50": lambda: cnn_embeddings(to3(ir)),
    }
    print("\nIllumination-invariance test (predict dark/bright scene; chance = 0.50):")
    for name, fn in feats.items():
        X, t = cached_features("thermal_" + name.replace(" ", "_"), fn)
        group = "thermal_vs_rgb"
        log_result(group, name, X, y, split, t, "RGB" if name.startswith("RGB") else "thermal")
    X_t, _ = cached_features("thermal_Thermal_ResNet50", feats["Thermal ResNet50"])
    plot_embedding_2d(X_t, y, names, "Thermal ResNet50 (PCA): no dark/bright structure", "thermal_pca.png")


if __name__ == "__main__":
    main()
