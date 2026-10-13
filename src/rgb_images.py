import time

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize

from common import (SEED, cached_features, describe, log_result, make_split,
                    plot_embedding_2d, save_fig)

N_PER_CLASS = 200
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

#only using a small subset form the entire dataset
def load_cifar10_subset(n_per_class=N_PER_CLASS, seed=SEED):
    import tensorflow as tf  # imported lazily so this file can be imported without TF
    (x, y), _ = tf.keras.datasets.cifar10.load_data()
    y = y.ravel()
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(y == c)[0], n_per_class, replace=False)
                          for c in range(10)])
    rng.shuffle(idx)
    return x[idx], y[idx]

def color_histogram_features(images, bins=16):
    feats = []
    for img in images:
        h = [np.histogram(img[..., c], bins=bins, range=(0, 256))[0] for c in range(3)]
        h = np.concatenate(h).astype(np.float32)
        feats.append(h / h.sum())
    return np.array(feats)

# using the Histogram of Oriented Gradients (Dalal & Triggs, 2005).
def hog_features(images, size=64, return_vis_for=None):
    feats, vis = [], None
    for i, img in enumerate(images):
        gray = resize(rgb2gray(img), (size, size), anti_aliasing=True)
        kw = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                  block_norm="L2-Hys", feature_vector=True)
        if return_vis_for is not None and i == return_vis_for:
            f, vis = hog(gray, visualize=True, **kw)
        else:
            f = hog(gray, **kw)
        feats.append(f)
    feats = np.array(feats, dtype=np.float32)
    return (feats, vis) if return_vis_for is not None else feats


def cnn_embeddings(images, batch_size=64):
    import tensorflow as tf
    base = tf.keras.applications.ResNet50(weights="imagenet", include_top=False,
                                          pooling="avg", input_shape=(224, 224, 3))
    out = []
    for i in range(0, len(images), batch_size):
        x = tf.image.resize(images[i:i + batch_size].astype("float32"), (224, 224))
        x = tf.keras.applications.resnet50.preprocess_input(x)  # ImageNet mean-subtraction
        out.append(base(x, training=False).numpy())
        print(f"    CNN batch {i // batch_size + 1}/{int(np.ceil(len(images) / batch_size))}", end="\r")
    print()
    return np.concatenate(out)

def main():
    print("== RGB IMAGES (CIFAR-10 subset) ==")
    X_img, y = load_cifar10_subset()
    describe("raw images (N,H,W,C)", X_img)  # a 4-D TENSOR: (batch, height, width, channels)
    split = make_split(y)

    fig, ax = plt.subplots(2, 10, figsize=(14, 3.2))
    for c in range(10):
        ax[0, c].imshow(X_img[np.where(y == c)[0][0]]);
        ax[0, c].set_title(CLASS_NAMES[c], fontsize=7)
        ax[1, c].imshow(X_img[np.where(y == c)[0][1]])
        ax[0, c].axis("off");
        ax[1, c].axis("off")
    plt.tight_layout();
    save_fig("rgb_samples.png")

    # --- features ------------------------------------------------------------
    X_hist, t_hist = cached_features("rgb_colorhist", lambda: color_histogram_features(X_img))
    X_hog, t_hog = cached_features("rgb_hog", lambda: hog_features(X_img))
    X_cnn, t_cnn = cached_features("rgb_resnet50", lambda: cnn_embeddings(X_img))
    for n, a in [("colour hist", X_hist), ("HOG", X_hog), ("ResNet50", X_cnn)]:
        describe(n, a)

    # --- HOG visualisation -----------------
    _, hog_img = hog_features(X_img[:1], return_vis_for=0)
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(X_img[0]);
    ax[0].set_title("input");
    ax[0].axis("off")
    ax[1].imshow(hog_img, cmap="gray");
    ax[1].set_title("HOG (dominant edge orientations)");
    ax[1].axis("off")
    plt.tight_layout();
    save_fig("rgb_hog_visual.png")

    # --- benchmark (same split + same linear probe for every method) ---------
    print("\nLinear-probe benchmark:")
    for name, X, t, note in [("Colour histogram", X_hist, t_hist, "hand-crafted"),
                             ("HOG", X_hog, t_hog, "hand-crafted"),
                             ("ResNet50 embedding", X_cnn, t_cnn, "pretrained CNN")]:
        log_result("rgb_image", name, X, y, split, t, note)
        plot_embedding_2d(X, y, CLASS_NAMES, f"{name} (PCA)", f"rgb_pca_{name.split()[0].lower()}.png")


if __name__ == "__main__":
    main()

