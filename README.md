# Multi-Modal Feature Extraction

**CS F320 Foundations of Data Science, Assignment 1** 

Feature extraction and comparison across images (RGB, multispectral, hyperspectral, thermal), video (RGB, plus a thermal motion pipeline), human speech and text, together with a hands-on look at tensors, XLA and TPU strategies in TensorFlow. Everything is Python and was developed on Google Colab.

The full write-up, with theory, code walk-throughs, comparison tables and a log of what went wrong, is in [`report/report.md`](report/report.md). This README covers how to run the code and what is in the repository.

## The idea in one paragraph

Each modality gets at least one traditional (hand-designed) and one modern (learned) representation. Every method is then scored under one fixed protocol: standardise the features, fit a logistic-regression *linear probe*, evaluate on the same held-out 30% split, and log accuracy, macro-F1, feature dimension and extraction time to `results/results.csv`. A linear classifier cannot invent structure, so its score measures the representation, not the classifier. For speech the split is by speaker, to stop a model from "recognising" voices instead of emotions.

## Repository layout

```
.
├── src/
│   ├── common.py             splits, linear probe, feature cache, result logging, plotting, downloader
│   ├── rgb_images.py         CIFAR-10: colour histogram, HOG, ResNet50
│   ├── spectral_images.py    Indian Pines (hyperspectral) and EuroSAT-MS (multispectral, --eurosat)
│   ├── thermal_features.py   thermal vs RGB on paired FLIR frames
│   ├── video_features.py     UCF101 subset: frame difference, optical-flow HOF, ResNet50; --thermal-dir demo
│   ├── audio_features.py     RAVDESS speech: log-mel, MFCC, prosody, wav2vec 2.0
│   ├── text_features.py      20 Newsgroups and IMDB: TF-IDF, GloVe, Sentence-BERT
│   └── tensors_tpu.py        tensors, sparse/ragged, bfloat16, XLA, TPUStrategy with fallback
├── results/
│   ├── results.csv           every benchmark number used in the report
│   └── figures/              plots produced by the scripts
├── report/
│   ├── report.md             the detailed report
│   └── figures/              figures embedded in the report
├── data/                     downloads and feature cache (git-ignored)
├── notebooks/
├── requirements.txt
└── README.md
```

## Running it on Colab

Use a GPU runtime for the CNN and transformer steps (Runtime, Change runtime type). Run from the repository root.

```bash
git clone https://github.com/UtkarshGupta0/fods-assignment-1.git
cd fods-assignment-1
pip install -q -r requirements.txt
```

Then run the scripts one at a time:

```bash
python src/rgb_images.py
python src/spectral_images.py            # Indian Pines only
python src/spectral_images.py --eurosat  # also EuroSAT-MS (about 2.1 GB download)
python src/thermal_features.py
python src/video_features.py
python src/audio_features.py
python src/text_features.py
python src/tensors_tpu.py
```

Each script prints the shape of what it extracts, writes figures to `results/figures/`, and upserts its rows into `results/results.csv`. Re-running a script overwrites its own rows instead of duplicating them.

**Optional flags**

| Command | Effect |
|---|---|
| `python src/spectral_images.py --eurosat --no-cnn` | skips the ResNet50 arm on EuroSAT (the slowest step) |
| `python src/thermal_features.py --folders RGB_DIR IR_DIR` | runs on any two folders of paired RGB and thermal images |
| `python src/video_features.py --thermal-dir FRAMES_DIR` | runs the MOG2 thermal-motion demo on a folder of sequential thermal frames |

### Caching and Colab disconnects

Feature matrices are cached in `data/cache/` along with their original extraction time, so a re-run after a disconnect takes seconds, not minutes. The cache is git-ignored, so back it up to Google Drive if a session matters:

```python
from google.colab import drive; drive.mount("/content/drive")
!cp -r data/cache/. /content/drive/MyDrive/fods_cache/      # save
!mkdir -p data && cp -rn /content/drive/MyDrive/fods_cache data/cache   # restore
```

### Pushing results back to GitHub

```bash
git config user.email "you@example.com" && git config user.name "Your Name"
git add src results report README.md requirements.txt
git commit -m "Add results"
git push
```

On Colab, store a fine-grained GitHub token (Contents: read and write, this repository only) in Colab Secrets as `GH_TOKEN` and clone with `https://<token>@github.com/<user>/<repo>.git`. Never commit the token.

## Datasets

All datasets are subsampled to keep the project runnable in a few hours on a free GPU. Seeds are fixed (42).

| Modality | Dataset | Subset used | Source |
|---|---|---|---|
| RGB image | CIFAR-10 | 2,000 images, 200 per class | `tf.keras.datasets` (auto-download) |
| Hyperspectral | Indian Pines (AVIRIS, 145×145×200) | all 10,249 labelled pixels, 16 classes | `.mat` files fetched from public GitHub mirrors; the original GIC-EHU host was unreachable |
| Multispectral | EuroSAT-MS (Sentinel-2, 13 bands) | 3,000 patches, 300 per class | Zenodo record 7711097, `EuroSAT_MS.zip` |
| Thermal + RGB | FLIR paired frames | 400 aligned pairs from the test split | Hugging Face `newguyme/flir_paired` |
| RGB video | UCF101, 10-class subset | 150 clips, 15 per class, 16 frames each | Hugging Face `sayakpaul/ucf101-subset` |
| Speech | RAVDESS, speech only | 1,440 clips, 24 actors, 8 emotions | Zenodo record 1188976, `Audio_Speech_Actors_01-24.zip` |
| Text | 20 Newsgroups (4 topics); IMDB reviews | 2,000 documents each | scikit-learn; Hugging Face `stanfordnlp/imdb` |

Datasets are not committed (`data/` is git-ignored). Every script downloads what it needs. If a download fails, `download_file` in `common.py` tries several mirrors and User-Agents, validates the result, and finally tells you which file to download by hand and where to put it (`data/`). Files already present and valid are used as they are.

Please respect each dataset's own license and citation requirements when reusing them.

## Results at a glance

Held-out accuracy of a linear probe. The full table, with macro-F1, dimensions and times, is in `results/results.csv` and the report.

| Modality | Traditional (best) | Learned (best) | Chance |
|---|---|---|---|
| RGB image | HOG 0.393 | ResNet50 0.862 | 0.10 |
| Hyperspectral | Raw spectrum 0.847 | 1D-CNN 0.925 | 0.0625 |
| Multispectral | 13-band stats + indices 0.861 | ResNet50 on RGB bands 0.948 | 0.10 |
| Text, 20 Newsgroups | TF-IDF 0.890 | Sentence-BERT 0.880 | 0.25 |
| Text, IMDB | TF-IDF 0.798 | GloVe 0.825 (Sentence-BERT 0.757) | 0.50 |
| Speech (speaker-independent) | MFCC + delta 0.440 | wav2vec 2.0 layer 6 0.685 | 0.125 |
| RGB video | Optical-flow HOF 0.356 | ResNet50 frame mean 0.911 | 0.10 |
| Thermal vs RGB | see report (the dark/bright test is confounded) | | 0.50 |

A few things worth knowing before you read the numbers:

- PCA with 20 components keeps 98.9% of the variance on Indian Pines yet scores 0.700, well below the raw spectrum, because variance is not the same as class information.
- Sentence-BERT lost the IMDB sentiment task. The report discusses likely reasons, none tested directly.
- Adding motion features did not help on the UCF101 subset: appearance dominates.
- Timings are measured on Colab sessions with different hardware, so compare them within a modality only.

## Tensors, XLA and TPUs

`src/tensors_tpu.py` needs no dataset. It covers tensors from rank 0 to rank 5 and the natural shape of every modality in this project, matrix operations and `tf.GradientTape`, sparse and ragged tensors, float32 versus bfloat16, XLA compilation with `jit_compile=True` (including a dump of the compiler's HLO), and a `get_strategy()` function that tries `TPUStrategy` and falls back to `MirroredStrategy` or the default strategy.

On the Colab runtime used, no TPU was attached, so the script fell back to a single Tesla T4 GPU. XLA made a small fused function about 1.5× faster (0.70 ms to 0.48 ms). **The TPU branch is implemented but was not executed on a TPU.**

## Limitations

- Small subsets and a single train/test split, with no repeated runs or confidence intervals. Differences of a few points are within noise (roughly ±0.5 for Indian Pines, ±1.4 text, ±2.4 audio, ±4 thermal, ±4.5 video).
- A linear probe can under-use very small or very wide feature sets.
- Thermal video was not evaluated on real data: the MOG2 pipeline in `video_features.py` was only tested on synthetic frames.
- The thermal-versus-RGB experiment uses a brightness label built from the RGB images, which is confounded by scene differences and by black borders on the dark frames.
- The random pixel split on Indian Pines makes the spatial-feature gain optimistic.
- CIFAR-10 images are upscaled from 32 to 224 pixels for ResNet50.
- Pretrained models were trained on data whose overlap with these benchmarks is unknown.

## Environment

Developed on Google Colab (Python 3.12 to 3.13, TensorFlow 2.x). The image models, the 1D-CNN and the TPU code use TensorFlow/Keras. The Hugging Face models (wav2vec 2.0, BERT) run on the PyTorch backend that Colab provides. Package list in `requirements.txt`.

## Acknowledgements and citations

Datasets: CIFAR-10 (Krizhevsky), Indian Pines (AVIRIS, Purdue/NASA), EuroSAT (Helber et al.), UCF101 (Soomro et al.), RAVDESS (Livingstone and Russo), 20 Newsgroups, IMDB reviews (Maas et al.), FLIR thermal data. Models: ResNet50 (He et al.), wav2vec 2.0 (Baevski et al.), BERT (Devlin et al.), Sentence-BERT / all-MiniLM-L6-v2 (Reimers and Gurevych), GloVe (Pennington et al.). Methods: HOG (Dalal and Triggs), Farnebäck optical flow, MOG2 background subtraction, Otsu thresholding.
