import os
import shutil
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
CACHE_DIR = os.path.join("data", "cache")
for _d in (RESULTS_DIR, FIG_DIR, CACHE_DIR):
    os.makedirs(_d, exist_ok=True)



# Splits
def make_split(y, groups=None, test_size=0.3, seed=SEED):
    idx = np.arange(len(y))
    if groups is None:
        return train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(idx, y, groups))



def cached_features(name, extractor_fn):
    path = os.path.join(CACHE_DIR, f"{name}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        X = z["X"].item() if z["sparse"] else z["X"]
        print(f"  [cache] loaded {name}  shape={X.shape}")
        return X, float(z["seconds"])
    t0 = time.perf_counter()
    X = extractor_fn()
    seconds = time.perf_counter() - t0
    is_sparse = sp.issparse(X)
    np.savez_compressed(path, X=np.array(X, dtype=object) if is_sparse else X,
                        seconds=seconds, sparse=is_sparse)
    return X, seconds

def linear_probe(X, y, split):
    tr, te = split
    # Sparse TF-IDF matrices must not be mean-centred (would destroy sparsity).
    scaler = StandardScaler(with_mean=not sp.issparse(X))
    clf = make_pipeline(scaler, LogisticRegression(max_iter=3000))
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    return accuracy_score(y[te], pred), f1_score(y[te], pred, average="macro")


def log_result(modality, method, X, y, split, extract_seconds, notes=""):
    acc, f1 = linear_probe(X, y, split)
    row = dict(modality=modality, method=method, feature_dim=X.shape[1],
               n_samples=X.shape[0], extract_seconds=round(extract_seconds, 2),
               accuracy=round(acc, 4), macro_f1=round(f1, 4), notes=notes)
    path = os.path.join(RESULTS_DIR, "results.csv")
    df = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        keep = ~((old.modality == modality) & (old.method == method))
        df = pd.concat([old[keep], df], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"  {modality:<14}{method:<28} dim={X.shape[1]:<6} "
          f"time={extract_seconds:7.1f}s  acc={acc:.3f}  macroF1={f1:.3f}")
    return row


def plot_embedding_2d(X, y, class_names, title, fname):

    if sp.issparse(X):
        Z = TruncatedSVD(2, random_state=SEED).fit_transform(X)
    else:
        Z = PCA(2, random_state=SEED).fit_transform(StandardScaler().fit_transform(X))
    plt.figure(figsize=(6, 5))
    for c, name in enumerate(class_names):
        m = y == c
        plt.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.6, label=name)
    plt.title(title)
    plt.legend(fontsize=6, markerscale=2, loc="best")
    plt.tight_layout()
    save_fig(fname)


def save_fig(fname):
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [figure] {path}")


def describe(name, arr):
    a = arr
    rng = "" if sp.issparse(a) else f" range=[{np.min(a):.3g}, {np.max(a):.3g}]"
    print(f"  {name}: shape={a.shape} dtype={a.dtype}{rng}")

def download_file(urls, dest):
    import shutil
    import urllib.request
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if isinstance(urls, str):
        urls = [urls]
    last = None
    for url in urls:
        try:
            print(f"  downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                                                                     "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
                return dest
        except Exception as e:
            last = e
            print(f"    failed: {e}")
            if os.path.exists(dest):
                os.remove(dest)
    raise RuntimeError(f"Could not download {os.path.basename(dest)} ({last}).\n"
                       f"Download it manually in your browser and upload it to '{dest}' "
                       f"(Colab: Files panel -> upload), then re-run.")


