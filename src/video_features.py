import glob
import os
import sys
import tarfile

import numpy as np
import matplotlib.pyplot as plt

from common import (cached_features, describe, log_result, make_split,
                    plot_embedding_2d, save_fig)

N_FRAMES = 16
VIDEOS_PER_CLASS = 15
FLOW_SIZE = 112


def download_ucf101_subset(dest="data/ucf101_subset"):
    if not glob.glob(f"{dest}/**/*.avi", recursive=True):
        from huggingface_hub import hf_hub_download
        os.makedirs(dest, exist_ok=True)
        path = hf_hub_download(repo_id="sayakpaul/ucf101-subset", filename="UCF101_subset.tar.gz",
                               repo_type="dataset")
        with tarfile.open(path) as t:
            t.extractall(dest)
    return sorted(glob.glob(f"{dest}/**/*.avi", recursive=True))


def class_of(path):
    parent = os.path.basename(os.path.dirname(path))
    if parent.lower() in ("train", "val", "test", "validation"):
        return os.path.basename(path).split("_")[1]
    return parent


def pick_videos(paths, per_class=VIDEOS_PER_CLASS, seed=42):
    rng = np.random.default_rng(seed)
    by = {}
    for p in paths:
        by.setdefault(class_of(p), []).append(p)
    names = sorted(by)
    chosen, labels = [], []
    for ci, n in enumerate(names):
        ps = sorted(by[n])
        for i in rng.choice(len(ps), min(per_class, len(ps)), replace=False):
            chosen.append(ps[i]);
            labels.append(ci)
    return chosen, np.array(labels), names


def read_frames(path, n_frames=N_FRAMES):
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = set(np.linspace(0, max(total - 1, 0), n_frames).astype(int).tolist())
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in idx:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    while len(frames) < n_frames:
        frames.append(frames[-1])
    return np.stack(frames[:n_frames])

def frame_diff_features(frames):
    import cv2
    g = np.stack([cv2.cvtColor(cv2.resize(f, (FLOW_SIZE, FLOW_SIZE)), cv2.COLOR_RGB2GRAY) for f in frames]).astype(
        np.float32)
    d = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))
    return np.array([d.mean(), d.std(), d.max(), *np.percentile(d, [25, 50, 75])], dtype=np.float32)


def hof_features(frames, bins=8, grid=2):
    import cv2
    g = [cv2.cvtColor(cv2.resize(f, (FLOW_SIZE, FLOW_SIZE)), cv2.COLOR_RGB2GRAY) for f in frames]
    acc, mags = np.zeros((grid, grid, bins), np.float32), []
    for a, b in zip(g[:-1], g[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        bin_idx = (ang / (2 * np.pi) * bins).astype(int) % bins
        h = FLOW_SIZE // grid
        for r in range(grid):
            for c in range(grid):
                m = mag[r * h:(r + 1) * h, c * h:(c + 1) * h].ravel()
                bi = bin_idx[r * h:(r + 1) * h, c * h:(c + 1) * h].ravel()
                acc[r, c] += np.bincount(bi, weights=m, minlength=bins)
        mags.append(mag.mean())
    acc /= max(len(g) - 1, 1)
    return np.concatenate([acc.ravel(), [np.mean(mags)]]).astype(np.float32)


def cnn_video_features(videos_frames):
    from rgb_images import cnn_embeddings
    n, t = len(videos_frames), videos_frames[0].shape[0]
    flat = np.concatenate(videos_frames)  # (N*T, H, W, 3)
    emb = cnn_embeddings(flat)  # (N*T, 2048)
    return emb.reshape(n, t, -1).mean(axis=1)



def thermal_motion_demo(frames_dir, max_frames=200):
    import cv2
    files = sorted(glob.glob(os.path.join(frames_dir, "*.png")) + glob.glob(os.path.join(frames_dir, "*.jpg"))
                   + glob.glob(os.path.join(frames_dir, "*.jpeg")))[:max_frames]
    if len(files) < 5:
        raise SystemExit(f"Need >=5 frames in {frames_dir}, found {len(files)}")
    bg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25, detectShadows=False)
    kernel = np.ones((3, 3), np.uint8)
    fg_frac, blobs, shown = [], [], []
    for i, f in enumerate(files):
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        mask = cv2.morphologyEx(bg.apply(im), cv2.MORPH_OPEN, kernel)
        n_lab, _ = cv2.connectedComponents(mask)
        fg_frac.append((mask > 0).mean());
        blobs.append(n_lab - 1)
        if i in (len(files) // 4, len(files) // 2, 3 * len(files) // 4):
            shown.append((im, mask))
    print(
        f"  {len(files)} frames | foreground fraction mean={np.mean(fg_frac):.4f} | mean blobs/frame={np.mean(blobs):.2f}")
    fig, ax = plt.subplots(2, len(shown), figsize=(4 * len(shown), 6))
    for k, (im, m) in enumerate(shown):
        ax[0, k].imshow(im, cmap="inferno");
        ax[0, k].set_title("thermal frame");
        ax[0, k].axis("off")
        ax[1, k].imshow(m, cmap="gray");
        ax[1, k].set_title("foreground mask (MOG2)");
        ax[1, k].axis("off")
    plt.tight_layout();
    save_fig("thermal_video_motion.png")


def main():
    if "--thermal-dir" in sys.argv:
        thermal_motion_demo(sys.argv[sys.argv.index("--thermal-dir") + 1])
        return
    print("== RGB VIDEO (UCF101 subset) ==")
    paths, y, names = pick_videos(download_ucf101_subset())
    print(f"  {len(paths)} clips, {len(names)} classes: {names}")
    frames = [read_frames(p) for p in paths]
    describe("one clip (T,H,W,C)", frames[0])
    split = make_split(y)

    X_diff, t1 = cached_features("video_framediff", lambda: np.array([frame_diff_features(f) for f in frames]))
    X_hof, t2 = cached_features("video_hof", lambda: np.array([hof_features(f) for f in frames]))
    X_cnn, t3 = cached_features("video_cnn_mean", lambda: cnn_video_features(frames))
    X_both = np.hstack([X_cnn, X_hof])
    for n, a in [("frame diff", X_diff), ("HOF", X_hof), ("CNN mean-pooled", X_cnn), ("CNN + HOF", X_both)]:
        describe(n, a)

    print("\nLinear-probe benchmark:")
    log_result("video_rgb", "Frame-difference stats", X_diff, y, split, t1, "6-d motion energy")
    log_result("video_rgb", "Optical-flow HOF", X_hof, y, split, t2, "motion only")
    log_result("video_rgb", "ResNet50 frame mean", X_cnn, y, split, t3, "appearance only")
    log_result("video_rgb", "ResNet50 + HOF", X_both, y, split, t3 + t2, "appearance + motion")
    plot_embedding_2d(X_cnn, y, names, "Video: CNN appearance (PCA)", "video_pca_cnn.png")


if __name__ == "__main__":
    main()
