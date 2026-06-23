"""FSDD wav -> log-mel 特征提取（Encoder 输入 / Decoder 输出 / 可视化统一格式）。"""

import torch


def normalize_feature_2d(feat):
    """逐样本 min-max -> [0,1]。"""
    lo, hi = feat.min(), feat.max()
    if (hi - lo) > 1e-6:
        return (feat - lo) / (hi - lo)
    return feat


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


def log_mel_from_wav(path, sample_rate, n_mels, n_frames, duration_sec, n_fft=512):
    """读取 wav -> resample -> pad/trim -> log-mel -> [n_mels, n_frames] in ~[0,1]。"""
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
    return normalize_feature_2d(logmel).float()


def audio_feature_shape(cfg):
    """返回 (n_mels, n_frames)。"""
    ac = cfg["audio"]
    return ac["n_mels"], ac["n_frames"]
