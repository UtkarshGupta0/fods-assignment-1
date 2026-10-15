import glob
import os
import zipfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common import (cached_features, describe, download_file, log_result, make_split,
                    plot_embedding_2d, save_fig)

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 64
ZIP_URL = ["https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1",
           "https://zenodo.org/api/records/1188976/files/Audio_Speech_Actors_01-24.zip/content",
           "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"]
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def download_ravdess(dest="data/ravdess"):
    if not glob.glob(f"{dest}/**/*.wav", recursive=True):
        os.makedirs(dest, exist_ok=True)
        zpath = download_file(ZIP_URL, "data/ravdess.zip", min_bytes=int(150e6), validate_zip=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest)
    return sorted(glob.glob(f"{dest}/**/*.wav", recursive=True))


def parse_labels(files):
    y = np.array([int(Path(f).stem.split("-")[2]) - 1 for f in files])
    actors = np.array([int(Path(f).stem.split("-")[6]) for f in files])
    return y, actors


def load_wave(path):
    import librosa
    w, _ = librosa.load(path, sr=SR, mono=True)
    w, _ = librosa.effects.trim(w, top_db=25)
    return w


def log_mel(w):
    import librosa
    S = librosa.feature.melspectrogram(y=w, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(S, ref=np.max)


def mfcc(w, n_mfcc=40):
    import librosa
    return librosa.feature.mfcc(y=w, sr=SR, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)


def stats(M):
    return np.concatenate([M.mean(1), M.std(1)])


def feat_logmel(files):
    return np.array([stats(log_mel(load_wave(f))) for f in files], dtype=np.float32)


def feat_mfcc(files):
    out = []
    for f in files:
        import librosa
        m = mfcc(load_wave(f))
        d = librosa.feature.delta(m)
        out.append(np.concatenate([stats(m), stats(d)]))
    return np.array(out, dtype=np.float32)


def prosody(w):
    import librosa
    f0 = librosa.yin(w, fmin=65, fmax=500, sr=SR, frame_length=1024, hop_length=HOP)
    rms = librosa.feature.rms(y=w, frame_length=N_FFT, hop_length=HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(w, frame_length=N_FFT, hop_length=HOP)[0]
    return np.array([f0.mean(), f0.std(), rms.mean(), rms.std(), zcr.mean(), zcr.std()])


def feat_mfcc_prosody(files, X_mfcc):
    P = np.array([prosody(load_wave(f)) for f in files], dtype=np.float32)
    return np.hstack([X_mfcc, P])



def feat_wav2vec(files, layers=(6, 12)):
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(dev).eval()
    out = []
    for i, f in enumerate(files):
        inp = fe(load_wave(f), sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            hs = model(inp.input_values.to(dev), output_hidden_states=True).hidden_states
        out.append(np.concatenate([hs[l][0].mean(0).cpu().numpy() for l in layers]))
        if i % 100 == 0:
            print(f"    wav2vec2 {i}/{len(files)}", end="\r")
    print()
    return np.array(out, dtype=np.float32)



def visualise_pipeline(path):
    import librosa, librosa.display
    w = load_wave(path)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(w, n_fft=N_FFT, hop_length=HOP)), ref=np.max)
    fig, ax = plt.subplots(4, 1, figsize=(8, 10))
    librosa.display.waveshow(w, sr=SR, ax=ax[0]);
    ax[0].set_title("1. Waveform (amplitude vs time)")
    librosa.display.specshow(D, sr=SR, hop_length=HOP, x_axis="time", y_axis="hz", ax=ax[1]);
    ax[1].set_title("2. STFT spectrogram (linear Hz)")
    librosa.display.specshow(log_mel(w), sr=SR, hop_length=HOP, x_axis="time", y_axis="mel", ax=ax[2]);
    ax[2].set_title("3. Log-mel spectrogram")
    librosa.display.specshow(mfcc(w), sr=SR, hop_length=HOP, x_axis="time", ax=ax[3]);
    ax[3].set_title("4. MFCCs (40 coefficients)")
    plt.tight_layout();
    save_fig("audio_pipeline.png")
    print(f"  waveform {w.shape}  ->  STFT {D.shape}  ->  log-mel {log_mel(w).shape}  ->  MFCC {mfcc(w).shape}")


def main():
    print("== AUDIO (RAVDESS speech) ==")
    files = download_ravdess()
    y, actors = parse_labels(files)
    print(f"  {len(files)} clips, {len(set(actors))} actors, classes: {EMOTIONS}")
    visualise_pipeline(files[0])


    split = make_split(y, groups=actors)

    X_mel, t1 = cached_features("audio_logmel", lambda: feat_logmel(files))
    X_mfcc, t2 = cached_features("audio_mfcc", lambda: feat_mfcc(files))
    X_mp, t3 = cached_features("audio_mfcc_prosody", lambda: feat_mfcc_prosody(files, X_mfcc))
    X_w2v, t4 = cached_features("audio_wav2vec", lambda: feat_wav2vec(files))
    for n, a in [("log-mel stats", X_mel), ("MFCC+delta stats", X_mfcc), ("MFCC+prosody", X_mp),
                 ("wav2vec2 L6+L12", X_w2v)]:
        describe(n, a)

    print("\nLinear-probe benchmark (speaker-independent split):")
    log_result("audio", "Log-mel stats", X_mel, y, split, t1, "128-d")
    log_result("audio", "MFCC + delta stats", X_mfcc, y, split, t2, "classical")
    log_result("audio", "MFCC + prosody", X_mp, y, split, t3 + t2, "classical + pitch/energy")
    log_result("audio", "wav2vec2 layer 6", X_w2v[:, :768], y, split, t4, "self-supervised transformer")
    log_result("audio", "wav2vec2 layer 12", X_w2v[:, 768:], y, split, t4, "last layer")
    plot_embedding_2d(X_mfcc, y, EMOTIONS, "MFCC features (PCA)", "audio_pca_mfcc.png")
    plot_embedding_2d(X_w2v[:, :768], y, EMOTIONS, "wav2vec2 layer 6 (PCA)", "audio_pca_wav2vec.png")


if __name__ == "__main__":
    main()
