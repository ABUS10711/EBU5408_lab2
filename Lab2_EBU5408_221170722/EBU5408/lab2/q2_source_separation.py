# q2_source_separation.py
# Author: <your name> | EBU5408 Lab 2 (Q2)

import numpy as np
import soundfile as sf
from pathlib import Path
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import librosa, librosa.display

################################################################################
# 0. Paths & Create Output Directories Automatically
################################################################################
BASE_DIR   = Path(__file__).parent             # lab2 directory
PREP_NPY   = BASE_DIR / "X_preprocessed.npy"   # Q1 output file
OUT_DIR    = BASE_DIR / "separated_sources"    # Separated audio output directory
SPEC_DIR   = BASE_DIR / "q1+q2images"          # Spectrogram output directory
OUT_DIR.mkdir(parents=True, exist_ok=True)     # Create if it doesn't exist
SPEC_DIR.mkdir(parents=True, exist_ok=True)    # Create if it doesn't exist

TARGET_SR      = 44100
MAX_ITERS      = 1000
TOL            = 1e-4
RANDOM_SEED    = 42

################################################################################
# 1. Load Preprocessed Data
################################################################################
X = np.load(PREP_NPY)               # shape=(n_channels, n_samples)
n_chan, n_samples = X.shape
print(f"Loaded whitened matrix: {n_chan} channels, {n_samples} samples")

################################################################################
# 2. Fast ICA Separation
################################################################################
ica = FastICA(
    # n_components=n_chan,            # Ignored when whiten=False, removed to suppress warning
    whiten=False,                   # Already whitened
    fun='logcosh',
    max_iter=MAX_ITERS,
    tol=TOL,
    random_state=RANDOM_SEED
)
S = ica.fit_transform(X.T).T        # shape=(n_components, n_samples)
print("FastICA converged:", ica.n_iter_ < MAX_ITERS, f"in {ica.n_iter_} iterations")

################################################################################
# 3. Write Separated Results
################################################################################
for i, sig in enumerate(S):
    sig = sig / (np.max(np.abs(sig)) + 1e-9) # Normalize
    out_path = OUT_DIR / f"source_{i+1}.wav"
    sf.write(out_path, sig.astype(np.float32), TARGET_SR)
    print("saved", out_path.name)

################################################################################
# 4. Visualization: Waveform & Spectrogram
################################################################################
time_axis = np.arange(n_samples) / TARGET_SR

# Waveform
fig_w, axes = plt.subplots(n_chan + n_chan, 1,
                           figsize=(10, 2*(n_chan + n_chan)),
                           sharex=True)
for c in range(n_chan):
    axes[c].plot(time_axis, X[c], linewidth=0.5)
    axes[c].set_title(f"Mixture channel {c+1}")
for k in range(n_chan):
    axes[n_chan+k].plot(time_axis, S[k], linewidth=0.5)
    axes[n_chan+k].set_title(f"Separated source {k+1}")
axes[-1].set_xlabel("Time (s)")
fig_w.tight_layout()
fig_w.savefig(SPEC_DIR / "out_waveforms.png", dpi=200)
plt.close(fig_w)
print(f"Waveform figure -> {SPEC_DIR.relative_to(BASE_DIR)}/out_waveforms.png")

# Spectrogram
def plot_spec(ax, sig, title):
    D = librosa.stft(sig, n_fft=1024, hop_length=256, window='hann')
    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(D), ref=np.max),
        sr=TARGET_SR,
        hop_length=256,
        y_axis='linear',  # <-- Changed 'log' to 'linear'
        x_axis='time',
        ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Frequency (Hz)") # Add Y-axis label
    ax.set_ylim(50, 12000)  # Set y-axis range to 50-12000 Hz

rows = 2 * n_chan
fig_s, axes = plt.subplots(rows, 1, figsize=(10, 2.2*rows), sharey=True) # Add sharey=True for consistent Y-axis range
for c in range(n_chan):
    plot_spec(axes[c], X[c], f"Mixture channel {c+1}")
for k in range(n_chan):
    plot_spec(axes[n_chan+k], S[k], f"Separated source {k+1}")
fig_s.tight_layout()
fig_s.savefig(SPEC_DIR / "out_spectrograms.png", dpi=200) # Modify output path
plt.close(fig_s)
print(f"Linear Spectrogram figure -> {SPEC_DIR.relative_to(BASE_DIR)}/out_spectrograms.png") # Modify print message
