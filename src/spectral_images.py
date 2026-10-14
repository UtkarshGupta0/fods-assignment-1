
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.ndimage import uniform_filter
from sklearn.decomposition import PCA

from common import (SEED, cached_features, describe, download_file, log_result,
                    make_split, plot_embedding_2d, save_fig)

DATA = "data"
GH = "https://raw.githubusercontent.com"
IP_URLS = {
    "Indian_pines_corrected.mat": [
        f"{GH}/gokriznastic/HybridSN/master/data/Indian_pines_corrected.mat",
        f"{GH}/eternal-vanguard/SpectralNET/master/data/Indian_pines_corrected.mat",
        f"{GH}/KonstantinosF/Classification-of-Hyperspectral-Image/master/Data/Indian_pines_corrected.mat",
        "https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"],
    "Indian_pines_gt.mat": [
        f"{GH}/gokriznastic/HybridSN/master/data/Indian_pines_gt.mat",
        f"{GH}/eternal-vanguard/SpectralNET/master/data/Indian_pines_gt.mat",
        f"{GH}/KonstantinosF/Classification-of-Hyperspectral-Image/master/Data/Indian_pines_gt.mat",
        "https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"],
}
IP_CLASSES = ["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees",
              "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill",
              "Soybean-clean", "Wheat", "Woods", "Bldg-Grass-Tree-Drives", "Stone-Steel-Towers"]



# PART A -- HYPERSPECTRAL

def load_indian_pines():

    for fname, urls in IP_URLS.items():
        download_file(urls, os.path.join(DATA, fname))
    cube = sio.loadmat(os.path.join(DATA, "Indian_pines_corrected.mat"))["indian_pines_corrected"]
    gt = sio.loadmat(os.path.join(DATA, "Indian_pines_gt.mat"))["indian_pines_gt"]
    return cube.astype(np.float32), gt.astype(np.int64)


def normalized_difference(cube, band_a, band_b):

    a, b = cube[..., band_a], cube[..., band_b]
    return (a - b) / (a + b + 1e-8)


def spectral_cnn_embeddings(X_spec, y, split, epochs=40):
    import tensorflow as tf
    tr, te = split
    mu, sd = X_spec[tr].mean(0), X_spec[tr].std(0) + 1e-6
    Xn = ((X_spec - mu) / sd)[..., None].astype("float32")
    tf.keras.utils.set_random_seed(SEED)
    inp = tf.keras.Input(shape=Xn.shape[1:])
    h = tf.keras.layers.Conv1D(32, 7, activation="relu")(inp)
    h = tf.keras.layers.MaxPool1D(2)(h)
    h = tf.keras.layers.Conv1D(64, 5, activation="relu")(h)
    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dropout(0.3)(h)
    emb = tf.keras.layers.Dense(64, activation="relu", name="embedding")(h)
    out = tf.keras.layers.Dense(int(y.max()) + 1, activation="softmax")(emb)
    model = tf.keras.Model(inp, out)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xn[tr], y[tr], epochs=epochs, batch_size=128, validation_split=0.1, verbose=0)
    print(f"    1D-CNN own test accuracy: {model.evaluate(Xn[te], y[te], verbose=0)[1]:.3f}")
    return tf.keras.Model(inp, emb).predict(Xn, batch_size=512, verbose=0)


def run_hyperspectral():
    print("== HYPERSPECTRAL: Indian Pines ==")
    cube, gt = load_indian_pines()
    describe("cube (H,W,bands)", cube)  # a 3-D tensor; bands = the 3rd axis
    H, W, B = cube.shape

    mask = gt > 0
    X_spec = cube[mask]
    y = gt[mask] - 1
    split = make_split(y)
    describe("pixel matrix X (N,bands)", X_spec)
    print(f"  labelled pixels: {len(y)}  classes: {len(np.unique(y))}")

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    rgb = cube[..., [29, 19, 9]]
    ax[0].imshow((rgb - rgb.min()) / (rgb.max() - rgb.min())); ax[0].set_title("false-colour composite")
    ax[1].imshow(gt, cmap="nipy_spectral"); ax[1].set_title("ground truth (16 classes)")
    im = ax[2].imshow(normalized_difference(cube, 41, 26), cmap="RdYlGn"); ax[2].set_title("NDVI-like index (approx. bands)")
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    for a in ax: a.axis("off")
    plt.tight_layout(); save_fig("hsi_overview.png")

    plt.figure(figsize=(7, 4))
    for c in [1, 5, 10, 13]:  # corn-notill, grass-trees, soybean-mintill, woods
        plt.plot(X_spec[y == c].mean(0), label=IP_CLASSES[c])
    plt.xlabel("band index (~400-2500 nm)"); plt.ylabel("reflectance (scaled)")
    plt.title("Spectral signatures: mean spectrum per class"); plt.legend(fontsize=7)
    plt.tight_layout(); save_fig("hsi_spectral_signatures.png")

    # --- feature sets -------------------------------------------------------
    tr = split[0]

    def f_pca():
        pca = PCA(20, random_state=SEED).fit(X_spec[tr])
        print(f"    PCA-20 explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        return pca.transform(X_spec).astype(np.float32)

    def f_spatial():
        pca = PCA(20, random_state=SEED).fit(X_spec[tr])
        pcs = pca.transform(cube.reshape(-1, B)).reshape(H, W, 20)
        neigh = uniform_filter(pcs, size=(5, 5, 1))[mask]
        return np.hstack([pca.transform(X_spec), neigh]).astype(np.float32)

    X_pca, t_pca = cached_features("hsi_pca20", f_pca)
    X_sp, t_sp = cached_features("hsi_spatial", f_spatial)
    X_cnn, t_cnn = cached_features("hsi_cnn1d", lambda: spectral_cnn_embeddings(X_spec, y, split))

    print("\nLinear-probe benchmark:")
    for name, X, t, note in [("Raw spectrum (200 bands)", X_spec, 0.0, "no extraction"),
                             ("PCA-20", X_pca, t_pca, "linear projection"),
                             ("PCA-20 + 5x5 neighbourhood", X_sp, t_sp, "spatial-spectral"),
                             ("1D-CNN embedding (64)", X_cnn, t_cnn, "learned; trained on train split")]:
        log_result("hyperspectral", name, X, y, split, t, note)
    plot_embedding_2d(X_pca, y, IP_CLASSES, "Indian Pines: PCA-20 features", "hsi_pca_embedding.png")
    plot_embedding_2d(X_cnn, y, IP_CLASSES, "Indian Pines: 1D-CNN features", "hsi_cnn_embedding.png")



# PART B -- MULTISPECTRAL (EuroSAT, Sentinel-2, 13 bands)
S2 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
IDX = {b: i for i, b in enumerate(S2)}
RGB_IDX = [IDX["B04"], IDX["B03"], IDX["B02"]]  # red, green, blue
EUROSAT_URLS = ["https://zenodo.org/records/7711096/files/EuroSAT_MS.zip?download=1",
                "https://zenodo.org/record/7711096/files/EuroSAT_MS.zip?download=1"]



def load_eurosat_ms(per_class=300, seed=SEED):
    import io
    import zipfile
    from collections import defaultdict
    try:
        import tifffile
    except ImportError:
        raise SystemExit("Missing package. Run: pip install -q tifffile imagecodecs")
    zpath = download_file(EUROSAT_URLS, os.path.join(DATA, "EuroSAT_MS.zip"))
    rng = np.random.default_rng(seed)
    X, y = [], []
    with zipfile.ZipFile(zpath) as z:
        by_class = defaultdict(list)
        for n in z.namelist():
            if n.lower().endswith(".tif") and "__MACOSX" not in n:
                by_class[os.path.basename(os.path.dirname(n))].append(n)
        names = sorted(by_class)
        for ci, cname in enumerate(names):
            files = sorted(by_class[cname])
            for i in rng.choice(len(files), min(per_class, len(files)), replace=False):
                img = tifffile.imread(io.BytesIO(z.read(files[i])))
                if img.shape[0] == 13:            # some readers return (bands, H, W)
                    img = np.moveaxis(img, 0, -1)
                X.append(img.astype(np.uint16)); y.append(ci)
    return np.stack(X), np.array(y), names


def band_descriptor(X, bands, lo, hi, nbins=10):
    feats = []
    for b in bands:
        v = X[..., b].reshape(len(X), -1).astype(np.float32)
        hist = np.stack([np.histogram(r, bins=nbins, range=(lo[b], hi[b]))[0] for r in v]) / v.shape[1]
        feats += [v.mean(1, keepdims=True), v.std(1, keepdims=True), hist]
    return np.hstack(feats)


def spectral_indices(X):
    f = lambda a, b: (X[..., IDX[a]].astype(np.float32) - X[..., IDX[b]]) / (
                X[..., IDX[a]].astype(np.float32) + X[..., IDX[b]] + 1e-6)
    out = []
    for idx in (f("B08", "B04"), f("B03", "B08"), f("B11", "B08")):
        flat = idx.reshape(len(X), -1)
        out += [flat.mean(1, keepdims=True), flat.std(1, keepdims=True)]
    return np.hstack(out)


def run_multispectral():
    print("== MULTISPECTRAL: EuroSAT-MS ==")
    X, y, names = load_eurosat_ms()
    describe("EuroSAT-MS (N,H,W,bands)", X)
    split = make_split(y)
    sub = X[:500].astype(np.float32)
    lo, hi = np.percentile(sub, 1, axis=(0, 1, 2)), np.percentile(sub, 99, axis=(0, 1, 2))

    X_rgb, t1 = cached_features("ms_rgb_stats", lambda: band_descriptor(X, RGB_IDX, lo, hi))
    X_all, t2 = cached_features("ms_all_stats", lambda: np.hstack([band_descriptor(X, range(13), lo, hi),
                                                                   spectral_indices(X)]))
    rows = [("RGB-only stats (3 bands)", X_rgb, t1, "same descriptor, 3 bands"),
            ("13-band stats + indices", X_all, t2, "adds NIR/red-edge/SWIR + NDVI/NDWI/NDBI")]

    if "--no-cnn" not in sys.argv:
        from rgb_images import cnn_embeddings
        def f_cnn():
            rgb = X[..., RGB_IDX].astype(np.float32)
            rgb = np.clip(rgb / np.percentile(rgb, 99) * 255, 0, 255).astype(np.uint8)
            return cnn_embeddings(rgb)

        X_cnn, t3 = cached_features("ms_resnet_rgb", f_cnn)
        rows.append(("ResNet50 on RGB bands", X_cnn, t3, "pretrained CNN, ignores non-visible bands"))

    print("\nLinear-probe benchmark:")
    for name, F, t, note in rows:
        log_result("multispectral", name, F, y, split, t, note)
    plot_embedding_2d(X_all, y, names, "EuroSAT: 13-band features (PCA)", "ms_pca_13band.png")


if __name__ == "__main__":
    run_hyperspectral()
    if "--eurosat" in sys.argv:
        run_multispectral()
