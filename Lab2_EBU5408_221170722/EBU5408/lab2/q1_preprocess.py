# q1_preprocess.py
import glob
import numpy as np
import soundfile as sf        # Read/write WAV files
import librosa.display        # Spectrogram display
import matplotlib.pyplot as plt # Plotting
from scipy import signal
from pathlib import Path
import os                     # For creating directories

# ---------- 1. Batch Loading ----------
AUDIO_DIR = Path("MysteryAudioLab2")


files = sorted(AUDIO_DIR.glob("*.wav"))
raw_signals, sr_list = [], []

for f in files:
    x, sr = sf.read(f)        # Read directly, keeping original channels
    raw_signals.append(x.T)   # Transpose to (channels, samples)
    sr_list.append(sr)

# ---------- 1.5. Generate and Save Raw Signal Spectrograms ----------
output_dir = "q1+q2images"
os.makedirs(output_dir, exist_ok=True) # Create output directory

print(f"Generating and saving raw signal images to '{output_dir}'...")
# Loop through all raw signals and corresponding original filenames
for i, (raw_signal, sr) in enumerate(zip(raw_signals, sr_list)):
    filename = files[i].stem # Get original filename (without extension)

    # If multi-channel, convert to mono for plotting
    if raw_signal.ndim > 1:
        plot_signal = np.mean(raw_signal, axis=0)
    else:
        plot_signal = raw_signal

    # Calculate STFT
    D = librosa.stft(plot_signal)
    # Convert to decibel scale
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Plot spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz') # Use original sample rate
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram of {filename} (Raw)')
    plt.tight_layout()

    # Save spectrogram
    output_path = os.path.join(output_dir, f"{filename}_raw_spectrogram.png")
    plt.savefig(output_path)
    plt.close() # Close the figure to free memory

print("Raw spectrogram generation complete.")


# ---------- 2. Unify Sample Rate ----------
# Check if all files have the same sample rate
unique_sr = set(sr_list)
if len(unique_sr) == 1:
    # If all files have the same sample rate
    target_sr = sr_list[0]
    print(f"All files have the same sample rate: {target_sr}Hz, keeping original sample rate")
    signals = raw_signals
else:
    # If sample rates differ, unify to the target sample rate
    # Can choose to unify to 44.1kHz, commonly used for high-quality audio
    target_sr = 44100  # Can also use 16000 for speech processing
    print(f"Detected different sample rates: {unique_sr}, unifying to {target_sr}Hz")
    signals = []
    for x, sr in zip(raw_signals, sr_list):
        if sr != target_sr:
            # Resample using librosa, handling multi-channel correctly if needed
            # librosa.resample expects (..., n_samples) or (n_channels, n_samples)
            # Our raw_signals are already (channels, samples) or (samples,)
            x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
        signals.append(x)

# ---------- 3. Convert to Mono, Remove DC Offset ----------
signals_mono = []
for x in signals:
    if x.ndim > 1:
        x = np.mean(x, axis=0)             # Can keep multi-channel if using multiple mics and algorithm supports it
    x = x - np.mean(x)                     # Remove DC offset
    signals_mono.append(x.astype(np.float32))

# ---------- 4. Amplitude Normalization ----------
norm_signals = [x / np.max(np.abs(x) + 1e-9) for x in signals_mono]

# ---------- 5. Pre-emphasis + Band-stop/Band-pass Filtering ----------
# Use a first-order high-pass filter (pre_emph=0.97) to boost high-frequency components
# Compensates for the natural decay of high-frequency energy in human voice
# Balances the speech spectrum (low-frequency energy in speech is often too high)
# Improves the signal-to-noise ratio for subsequent analysis
# Band-pass filtering: Remove high-frequency noise, retain main frequency components of voice and music
pre_emph = 0.97
bp_lo, bp_hi = 50, 12000                    # Adjusted to a wider band to include low and high frequencies of music
proc_signals = []
b, a = signal.butter(4, [bp_lo, bp_hi], btype='bandpass', fs=target_sr)

for x in norm_signals:
    x = signal.lfilter([1, -pre_emph], 1, x)   # Pre-emphasis
    x = signal.filtfilt(b, a, x)               # Zero-phase band-pass filter
    proc_signals.append(x)


# ---------- 7. Whitening (Preparation for ICA/PCA) ----------
# Align all channels to the same length, then whiten
min_len = min(map(len, proc_signals))
X = np.stack([x[:min_len] for x in proc_signals])  # shape=(n_channels, n_samples)

# Zero mean
X -= X.mean(axis=1, keepdims=True)

# Whitening
cov = np.cov(X)
d, E = np.linalg.eigh(cov) # Eigenvalue decomposition of the covariance matrix
D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-5)) # Inverse square root of eigenvalues (add small epsilon for stability)
X_white = D_inv_sqrt @ E.T @ X # Whitening transformation: project data onto eigenvectors and scale by inverse sqrt eigenvalues

np.save("X_preprocessed.npy", X_white)      # To be called by Q2
print("Saved whitened matrix with shape", X_white.shape)

# ---------- 8. Generate and Save Spectrograms ----------


print(f"Generating and saving preprocessed signal images to '{output_dir}'...")
# Loop through all processed signals and corresponding original filenames
for i, signal_to_plot in enumerate(proc_signals):
    filename = files[i].stem # Get original filename (without extension)

    # Calculate STFT
    D = librosa.stft(signal_to_plot)
    # Convert to decibel scale
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Plot spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=target_sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram of {filename} (Preprocessed)')
    plt.tight_layout()

    # Save spectrogram
    output_path = os.path.join(output_dir, f"{filename}_preprocessed_spectrogram.png") # Modify filename to distinguish
    plt.savefig(output_path)
    plt.close() # Close the figure to free memory

print("Preprocessed spectrogram generation complete.")
