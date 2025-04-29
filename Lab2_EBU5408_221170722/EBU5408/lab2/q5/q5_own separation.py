import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy import signal
from sklearn.decomposition import FastICA
from pathlib import Path
import os

# 参数设置
TARGET_SR = 44100  # 目标采样率
PRE_EMPH = 0.97  # 预加重系数
BP_LO = 50  # 带通滤波低频截止
BP_HI = 12000  # 带通滤波高频截止
MAX_ITERS = 1000  # ICA 最大迭代次数
TOL = 1e-4  # ICA 收敛容差
RANDOM_SEED = 42  # 随机种子
N_COMPONENTS = 4  # 分离源数量

# 路径设置
AUDIO_DIR = Path(".")
OUTPUT_DIR = Path("q5_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # 创建输出目录

# 音频文件
audio_files = [
    AUDIO_DIR / "microphone1.wav",
    AUDIO_DIR / "microphone2.wav",
    AUDIO_DIR / "microphone3.wav",
    AUDIO_DIR / "microphone4.wav"
]


# -------------------------- 预处理函数 --------------------------

def preprocess_signals(files, target_sr=TARGET_SR, pre_emph=PRE_EMPH, bp_lo=BP_LO, bp_hi=BP_HI):
    # 1. 加载音频
    raw_signals, sr_list = [], []
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"Audio file not found: {f}")
        x, sr = sf.read(f)
        if x.ndim > 2:
            raise ValueError(f"Audio file {f} has more than 2 dimensions: {x.shape}")
        if x.ndim == 1:
            x = x[:, None]  # 转换为 (samples, 1)
        raw_signals.append(x.T)  # 转成 (channels, samples)
        sr_list.append(sr)

    # 2. 保存原始信号频谱图
    for i, (raw_signal, sr, f) in enumerate(zip(raw_signals, sr_list, files)):
        filename = f.stem
        plot_signal = np.mean(raw_signal, axis=0) if raw_signal.shape[0] > 1 else raw_signal[0]
        D = librosa.stft(plot_signal)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram of {filename} (Raw)')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{filename}_raw_spectrogram.png")
        plt.close()
    print(f"Raw spectrograms saved to {OUTPUT_DIR}")

    # 3. 统一采样率
    signals = []
    for x, sr in zip(raw_signals, sr_list):
        if sr != target_sr:
            x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
        signals.append(x)

    # 4. 转单声道、去直流偏移
    signals_mono = []
    for x in signals:
        if x.shape[0] > 1:
            x = np.mean(x, axis=0)  # 转单声道
        elif x.shape[0] == 1:
            x = x[0]  # 展平
        x = x - np.mean(x)  # 去 DC
        signals_mono.append(x.astype(np.float32))

    # 5. 幅度归一化
    norm_signals = [x / np.max(np.abs(x) + 1e-9) for x in signals_mono]

    # 6. 预加重 + 带通滤波
    proc_signals = []
    nyquist = target_sr / 2
    if bp_hi >= nyquist:
        bp_hi = nyquist - 100  # 确保不超过 Nyquist 频率
    b, a = signal.butter(4, [bp_lo, bp_hi], btype='bandpass', fs=target_sr)
    for x in norm_signals:
        x = signal.lfilter([1, -pre_emph], 1, x)  # 预加重
        x = signal.filtfilt(b, a, x)  # 带通滤波
        proc_signals.append(x)

    # 7. 保存预处理信号频谱图
    for i, (signal_to_plot, f) in enumerate(zip(proc_signals, files)):
        filename = f.stem
        D = librosa.stft(signal_to_plot)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_db, sr=target_sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram of {filename} (Preprocessed)')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{filename}_preprocessed_spectrogram.png")
        plt.close()
    print(f"Preprocessed spectrograms saved to {OUTPUT_DIR}")

    # 8. 白化
    min_len = min(map(len, proc_signals))
    X = np.stack([x[:min_len] for x in proc_signals])  # shape=(n_channels, n_samples)
    if X.ndim != 2:
        raise ValueError(f"X is not 2D: {X.shape}")
    X -= X.mean(axis=1, keepdims=True)
    cov = np.cov(X)
    d, E = np.linalg.eigh(cov)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-5))
    X_white = D_inv_sqrt @ E.T @ X

    return X_white, proc_signals, target_sr


# -------------------------- ICA 分离函数 --------------------------

def separate_sources(X, n_components=N_COMPONENTS, max_iter=MAX_ITERS, tol=TOL, random_seed=RANDOM_SEED):
    # 应用 FastICA
    ica = FastICA(
        whiten=False,  # 数据已白化
        fun='logcosh',
        max_iter=max_iter,
        tol=tol,
        random_state=random_seed
    )
    S = ica.fit_transform(X.T).T  # shape=(n_components, n_samples)
    print(f"FastICA converged: {ica.n_iter_ < max_iter} in {ica.n_iter_} iterations")
    return S


# -------------------------- 可视化函数 --------------------------

def plot_waveforms_and_spectrograms(X, S, target_sr, output_dir, n_channels, n_components):
    time_axis = np.arange(X.shape[1]) / target_sr
    # 波形图
    fig_w, axes = plt.subplots(n_channels + n_components, 1, figsize=(10, 2 * (n_channels + n_components)), sharex=True)
    for c in range(n_channels):
        axes[c].plot(time_axis, X[c], linewidth=0.5)
        axes[c].set_title(f"Mixture channel {c + 1}")
    for k in range(n_components):
        axes[n_channels + k].plot(time_axis, S[k], linewidth=0.5)
        axes[n_channels + k].set_title(f"Separated source {k + 1}")
    axes[-1].set_xlabel("Time (s)")
    fig_w.tight_layout()
    fig_w.savefig(output_dir / "out_waveforms.png", dpi=200)
    plt.close(fig_w)
    print(f"Waveform figure saved to {output_dir}/out_waveforms.png")

    # 频谱图
    def plot_spec(ax, sig, title):
        D = librosa.stft(sig, n_fft=1024, hop_length=256, window='hann')
        librosa.display.specshow(
            librosa.amplitude_to_db(np.abs(D), ref=np.max),
            sr=target_sr, hop_length=256, y_axis='linear', x_axis='time', ax=ax
        )
        ax.set_title(title)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(BP_LO, BP_HI + 1000)

    fig_s, axes = plt.subplots(n_channels + n_components, 1, figsize=(10, 2.2 * (n_channels + n_components)),
                               sharey=True)
    for c in range(n_channels):
        plot_spec(axes[c], X[c], f"Mixture channel {c + 1}")
    for k in range(n_components):
        plot_spec(axes[n_channels + k], S[k], f"Separated source {k + 1}")
    fig_s.tight_layout()
    fig_s.savefig(output_dir / "out_spectrograms.png", dpi=200)
    plt.close(fig_s)
    print(f"Spectrogram figure saved to {output_dir}/out_spectrograms.png")


# -------------------------- 主执行逻辑 --------------------------

def main():
    print(f"Processing audio files: {[f.name for f in audio_files]}")

    # 预处理
    X_white, proc_signals, target_sr = preprocess_signals(audio_files)
    print(f"Whitened matrix shape: {X_white.shape}")

    # ICA 分离
    S = separate_sources(X_white, n_components=N_COMPONENTS)

    # 保存分离音频
    for i, sig in enumerate(S):
        sig = sig / (np.max(np.abs(sig)) + 1e-9)  # 归一化
        out_path = OUTPUT_DIR / f"source_{i + 1}.wav"
        sf.write(out_path, sig.astype(np.float32), target_sr)
        print(f"Saved separated source: {out_path}")

    # 可视化
    plot_waveforms_and_spectrograms(X_white, S, target_sr, OUTPUT_DIR, n_channels=len(audio_files),
                                    n_components=N_COMPONENTS)


if __name__ == "__main__":
    main()