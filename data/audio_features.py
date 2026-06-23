"""FSDD wav -> log-mel 特征提取（Encoder 输入 / Decoder 输出 / 可视化统一格式）。"""

import glob
import os

import torch

from paths import resolve_from_root


def normalize_feature_per_sample(feat):
    """逐样本 min-max -> [0,1]（legacy v3/v4）。"""
    lo, hi = feat.min(), feat.max()
    if (hi - lo) > 1e-6:
        return (feat - lo) / (hi - lo)
    return feat


def normalize_feature_global(feat, lo, hi):
    """全局固定缩放 -> clamp [0,1]。"""
    if (hi - lo) > 1e-6:
        out = (feat - lo) / (hi - lo)
    else:
        out = feat - lo
    return out.clamp(0.0, 1.0)


def _load_wav_mono(path):
    """读取单声道 wav；优先 soundfile，兼容 torchaudio 多版本。"""
    try:
        import soundfile as sf
        data, sr = sf.read(path, always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return torch.from_numpy(data).float().unsqueeze(0), int(sr)
    except Exception:
        pass

    import torchaudio
    try:
        wav, sr = torchaudio.load(path, backend="soundfile")
    except TypeError:
        wav, sr = torchaudio.load(path)
    except Exception:
        wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, int(sr)


def log_mel_raw(path, sample_rate, n_mels, n_frames, duration_sec, n_fft=512):
    """读取 wav -> log-mel（未做归一化）。"""
    import torchaudio

    wav, sr = _load_wav_mono(path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    target_len = int(sample_rate * duration_sec)
    if wav.size(1) < target_len:
        wav = torch.nn.functional.pad(wav, (0, target_len - wav.size(1)))
    else:
        wav = wav[:, :target_len]

    hop = max(1, target_len // n_frames)
    mel_t = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
    )
    mel = mel_t(wav).squeeze(0)
    logmel = torch.log1p(mel)
    if logmel.size(1) < n_frames:
        logmel = torch.nn.functional.pad(logmel, (0, n_frames - logmel.size(1)))
    else:
        logmel = logmel[:, :n_frames]
    return logmel.float()


def log_mel_from_wav(path, sample_rate, n_mels, n_frames, duration_sec,
                     n_fft=512, norm_mode="global", norm_stats=None):
    """读取 wav -> log-mel -> [n_mels, n_frames] in ~[0,1]。"""
    logmel = log_mel_raw(path, sample_rate, n_mels, n_frames, duration_sec, n_fft)
    if norm_mode == "per_sample":
        return normalize_feature_per_sample(logmel)
    if norm_stats is None:
        raise ValueError("norm_mode=global 需要 norm_stats（lo/hi）。")
    return normalize_feature_global(logmel, norm_stats["lo"], norm_stats["hi"])


def audio_feature_shape(cfg):
    """返回 (n_mels, n_frames)。"""
    ac = cfg["audio"]
    return ac["n_mels"], ac["n_frames"]


def _fsdd_train_wav_paths(cfg):
    """FSDD 训练划分 wav 路径（与 dataset 划分一致）。"""
    from data.fsdd import ensure_fsdd, fsdd_recordings_dir

    def _parse(name):
        parts = os.path.splitext(os.path.basename(name))[0].split("_")
        if len(parts) < 3:
            return None
        try:
            return int(parts[-1])
        except ValueError:
            return None

    ac = cfg["audio"]
    rec = fsdd_recordings_dir(cfg)
    if not glob.glob(os.path.join(rec, "*.wav")) and ac.get("auto_download", True):
        ensured = ensure_fsdd(cfg)
        if ensured:
            rec = ensured
    paths = []
    for f in sorted(glob.glob(os.path.join(rec, "*.wav"))):
        idx = _parse(f)
        if idx is not None and idx >= 5:
            paths.append(f)
    return paths


def compute_audio_norm_stats(cfg):
    """在训练集 wav 上统计 log-mel 全局分位数（默认 p1–p99）。"""
    ac = cfg["audio"]
    n_mels, n_frames = audio_feature_shape(cfg)
    p_lo = float(ac.get("norm_percentile_lo", 1.0))
    p_hi = float(ac.get("norm_percentile_hi", 99.0))
    paths = _fsdd_train_wav_paths(cfg)
    if not paths:
        raise FileNotFoundError("无法统计 audio norm：未找到 FSDD 训练 wav。")

    chunks = []
    for f in paths:
        chunks.append(
            log_mel_raw(f, ac["sample_rate"], n_mels, n_frames, ac["duration_sec"]).flatten()
        )
    allv = torch.cat(chunks)
    lo = torch.quantile(allv, p_lo / 100.0).item()
    hi = torch.quantile(allv, p_hi / 100.0).item()
    if hi <= lo:
        hi = lo + 1e-6
    return {
        "lo": lo, "hi": hi,
        "n_mels": n_mels, "n_frames": n_frames,
        "percentile_lo": p_lo, "percentile_hi": p_hi,
        "n_wavs": len(paths),
    }


def save_audio_norm_stats(stats, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(stats, path)


def load_audio_norm_stats(path):
    return torch.load(path, map_location="cpu")


def ensure_audio_norm_stats(cfg):
    """加载或计算全局 norm stats；shape 变化时自动重算。"""
    ac = cfg["audio"]
    if ac.get("norm_mode", "global") == "per_sample":
        return None

    stats_path = str(resolve_from_root(
        ac.get("norm_stats_path", "_data/audio_norm_stats.pt")))
    n_mels, n_frames = audio_feature_shape(cfg)

    if os.path.isfile(stats_path):
        stats = load_audio_norm_stats(stats_path)
        if (stats.get("n_mels") == n_mels
                and stats.get("n_frames") == n_frames):
            return stats
        print(f"[audio] norm stats 尺寸不匹配，重新统计 -> {stats_path}", flush=True)

    print(f"[audio] 统计全局 log-mel norm（train wav）...", flush=True)
    stats = compute_audio_norm_stats(cfg)
    save_audio_norm_stats(stats, stats_path)
    print(f"[audio] norm stats: lo={stats['lo']:.4f} hi={stats['hi']:.4f} "
          f"({stats['n_wavs']} wavs) -> {stats_path}", flush=True)
    return stats
