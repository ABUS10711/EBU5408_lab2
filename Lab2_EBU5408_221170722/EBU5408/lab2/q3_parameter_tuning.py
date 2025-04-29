import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import librosa
import os
from scipy import signal
from pathlib import Path

# Load preprocessed data
TARGET_SR = 44100  # Target sampling rate

# Standard parameters
standard_params = {
    'n_components': 3,  # Number of components
    'max_iter': 5,      # Maximum iterations
    'tol': 1e-4,        # Convergence tolerance
    'fun': 'logcosh',   # G function
    'bp_lo': 50,        # Bandpass filter lower frequency limit
    'bp_hi': 10000      # Bandpass filter upper frequency limit
}

# Parameter options (change one parameter at a time)
param_options = {
    'n_components': [2, 3],
    'max_iter': [1, 2, 5],
    'tol': [0.1, 1e-4],
    'bp_lo': [50, 80],
    'bp_hi': [8000, 10000]
}

# Create folders to save images and audio
tuning_results_dir = "q3_tuning_results"
os.makedirs(tuning_results_dir, exist_ok=True)

# -------------------------- Preprocessing Function --------------------------

def preprocess_signals(files, target_sr=TARGET_SR, pre_emph=0.97, bp_lo=50, bp_hi=10000):
    # Read audio and preprocess
    raw_signals, sr_list = [], []
    for f in files:
        x, sr = sf.read(f)  # Read directly, keep original channels
        raw_signals.append(x.T)  # Transpose to (channels, samples)
        sr_list.append(sr)

    # Unify sampling rate
    unique_sr = set(sr_list)
    if len(unique_sr) == 1:
        target_sr = sr_list[0] # Use the common sampling rate if all files have the same one
    signals = []
    for x, sr in zip(raw_signals, sr_list):
        if sr != target_sr:
            x = librosa.resample(x, orig_sr=sr, target_sr=target_sr) # Resample
        signals.append(x)

    # Convert to mono, remove DC offset
    signals_mono = []
    for x in signals:
        if x.ndim > 1:
            x = np.mean(x, axis=0) # Convert to mono
        x = x - np.mean(x)  # Remove DC component
        signals_mono.append(x.astype(np.float32))

    # Amplitude normalization
    norm_signals = [x / np.max(np.abs(x) + 1e-9) for x in signals_mono] # Normalize to [-1, 1]

    # Bandpass filtering
    proc_signals = []
    b, a = signal.butter(4, [bp_lo, bp_hi], btype='bandpass', fs=target_sr) # 4th order Butterworth bandpass filter
    for x in norm_signals:
        x = signal.lfilter([1, -pre_emph], 1, x)  # Pre-emphasis
        x = signal.filtfilt(b, a, x)  # Bandpass filter (zero-phase filtering)
        proc_signals.append(x)

    # Whitening
    min_len = min(map(len, proc_signals)) # Get the shortest signal length
    X = np.stack([x[:min_len] for x in proc_signals])  # Stack signals, shape=(n_channels, n_samples)
    X -= X.mean(axis=1, keepdims=True)  # Zero-mean
    cov = np.cov(X) # Calculate covariance matrix
    d, E = np.linalg.eigh(cov) # Eigenvalue decomposition
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-5)) # Calculate D^(-1/2)
    X_white = D_inv_sqrt @ E.T @ X # Whitening

    return X_white

# -------------------------- ICA Parameter Tuning and Evaluation --------------------------

def evaluate_separation(files, n_components, max_iter, tol, fun, bp_lo, bp_hi, result_folder):
    # Reprocess audio each time with current filter parameters
    X_white = preprocess_signals(files, bp_lo=bp_lo, bp_hi=bp_hi)

    # Initialize and fit FastICA model
    ica = FastICA(n_components=n_components, max_iter=max_iter, tol=tol, fun=fun, random_state=42)
    S = ica.fit_transform(X_white.T).T # Separated source signals, transpose back to (n_components, n_samples)
    A_est = ica.mixing_ # Estimated mixing matrix

    # Save separated audio
    for i in range(n_components):
        sf.write(os.path.join(result_folder,
                              f"source_{i + 1}_n_components_{n_components}_bp_lo_{bp_lo}_bp_hi_{bp_hi}_max_iter_{max_iter}_tol_{tol}_fun_{fun}.wav"),
                 S[i], TARGET_SR)

    # Plot waveforms
    fig, axes = plt.subplots(n_components, 1, figsize=(10, 2 * (n_components)), sharex=True)
    for k in range(n_components):
        axes[k].plot(np.arange(S.shape[1]) / TARGET_SR, S[k], linewidth=0.5)
        axes[k].set_title(f"Separated source {k + 1}")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()

    # Save waveform plot
    fig_waveform_path = os.path.join(result_folder,
                                     f"waveform_n_components_{n_components}_bp_lo_{bp_lo}_bp_hi_{bp_hi}_max_iter_{max_iter}_tol_{tol}_fun_{fun}.png")
    fig.savefig(fig_waveform_path, dpi=200)
    plt.close(fig)

    # Plot spectrograms (linear y-axis)
    fig_s, axes_s = plt.subplots(n_components + X_white.shape[0], 1,
                                 figsize=(10, 2 * (n_components + X_white.shape[0])), sharex=True)

    # Plot spectrograms of original mixture signals
    for c in range(X_white.shape[0]):
        D = librosa.stft(X_white[c], n_fft=1024, hop_length=256, window='hann') # Short-time Fourier transform
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max), # Amplitude to dB
                                 sr=TARGET_SR,
                                 hop_length=256,
                                 y_axis='linear', # Linear frequency axis
                                 x_axis='time',
                                 ax=axes_s[c])
        axes_s[c].set_title(f"Mixture channel {c + 1}")
        axes_s[c].set_ylim(0, bp_hi + 1000) # Set y-axis range

    # Plot spectrograms of separated signals
    for k in range(n_components):
        D = librosa.stft(S[k], n_fft=1024, hop_length=256, window='hann')
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max),
                                 sr=TARGET_SR,
                                 hop_length=256,
                                 y_axis='linear',
                                 x_axis='time',
                                 ax=axes_s[X_white.shape[0] + k])
        axes_s[X_white.shape[0] + k].set_title(f"Separated source {k + 1}")
        axes_s[X_white.shape[0] + k].set_ylim(0, bp_hi + 1000)

    axes_s[-1].set_xlabel("Time (s)")
    fig_s.tight_layout()

    # Save spectrogram plot
    fig_spectrogram_path = os.path.join(result_folder,
                                        f"spectrogram_n_components_{n_components}_bp_lo_{bp_lo}_bp_hi_{bp_hi}_max_iter_{max_iter}_tol_{tol}_fun_{fun}.png")
    fig_s.savefig(fig_spectrogram_path, dpi=200)
    plt.close(fig_s)

# Tune parameters and evaluate
AUDIO_DIR = "MysteryAudioLab2" # Audio file directory
files = sorted(Path(AUDIO_DIR).glob("*.wav")) # Get all wav files

# Test standard parameter combination
result_folder = os.path.join(tuning_results_dir, "standard") # Standard parameters result folder
os.makedirs(result_folder, exist_ok=True)
print(f"Evaluating standard parameters: {standard_params}")
evaluate_separation(
    files,
    n_components=standard_params['n_components'],
    max_iter=standard_params['max_iter'],
    tol=standard_params['tol'],
    fun=standard_params['fun'],
    bp_lo=standard_params['bp_lo'],
    bp_hi=standard_params['bp_hi'],
    result_folder=result_folder
)

# Evaluate by changing one parameter at a time
for param_name, options in param_options.items():
    for value in options:
        params = standard_params.copy() # Copy standard parameters
        params[param_name] = value # Modify the current parameter to tune
        # Create corresponding result folder
        result_folder = os.path.join(tuning_results_dir, param_name, str(value))
        os.makedirs(result_folder, exist_ok=True)
        print(f"Evaluating {param_name}={value}, other parameters: {params}")
        # Evaluate using the current parameter combination
        evaluate_separation(
            files,
            n_components=params['n_components'],
            max_iter=params['max_iter'],
            tol=params['tol'],
            fun=params['fun'],
            bp_lo=params['bp_lo'],
            bp_hi=params['bp_hi'],
            result_folder=result_folder
        )
