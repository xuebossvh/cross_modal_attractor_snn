"""配对 (图像, 音频, 标签) 数据集 —— MNIST + FSDD 真实语音（log-mel）。

音频特征：2D log-mel [n_mels, n_frames]，归一化到 ~[0,1]（global 或 per_sample）。
Audio Encoder 输入、Audio Decoder 输出、audio recovery loss 均使用此格式。
"""

import csv
import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from data.audio_features import (
    log_mel_from_wav, audio_feature_shape, ensure_audio_norm_stats,
)
from data.fsdd import ensure_fsdd, fsdd_recordings_dir
from paths import resolve_from_root


class _SyntheticImages:
    def __init__(self, num_samples, num_classes, seed=0):
        rng = np.random.default_rng(seed)
        self.labels = rng.integers(0, num_classes, size=num_samples)
        self.num_classes = num_classes
        self._rng = rng

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        label = int(self.labels[i])
        img = np.zeros((28, 28), dtype=np.float32)
        cx = 6 + (label % 5) * 4
        cy = 6 + (label // 5) * 12
        yy, xx = np.mgrid[0:28, 0:28]
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / 18.0))
        img += blob.astype(np.float32)
        img += 0.05 * self._rng.standard_normal((28, 28)).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)
        return torch.from_numpy(img).unsqueeze(0), label


def _make_audio_prototypes(num_classes, n_mels, n_frames, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(num_classes, n_mels, n_frames, generator=g)


def _parse_fsdd_name(path):
    """解析 FSDD 文件名 {digit}_{speaker}_{index}.wav -> (digit, index) 或 None。"""
    name = os.path.splitext(os.path.basename(path))[0]
    parts = name.split("_")
    if len(parts) < 3:
        return None
    try:
        digit = int(parts[0])
        idx = int(parts[-1])
    except ValueError:
        return None
    return digit, idx


def _load_fsdd_by_digit(cfg, train):
    """加载 FSDD log-mel，返回特征池及与其同序的 wav 路径池。"""
    ac = cfg["audio"]
    try:
        import torchaudio  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "缺少 torchaudio，无法加载 FSDD 真实音频。"
            "请执行: pip install torchaudio soundfile"
        ) from e

    rec = fsdd_recordings_dir(cfg)
    if not glob.glob(os.path.join(rec, "*.wav")):
        if ac.get("auto_download", True):
            ensured = ensure_fsdd(cfg)
            if ensured:
                rec = ensured
    files = sorted(glob.glob(os.path.join(rec, "*.wav")))
    if not files:
        raise FileNotFoundError(
            f"未找到 FSDD wav 文件: {rec}\n"
            "请设 audio.auto_download: true，或手动下载 FSDD 解压到该目录。"
        )

    n_mels, n_frames = audio_feature_shape(cfg)
    norm_mode = ac.get("norm_mode", "global")
    norm_stats = cfg.get("_audio_norm_stats")
    by_digit = {d: [] for d in range(cfg["dims"]["num_classes"])}
    paths_by_digit = {d: [] for d in range(cfg["dims"]["num_classes"])}
    n_parse_skip = 0
    n_split_skip = 0
    mel_errors = []

    for f in files:
        parsed = _parse_fsdd_name(f)
        if parsed is None:
            n_parse_skip += 1
            continue
        digit, idx = parsed
        if digit not in by_digit:
            n_parse_skip += 1
            continue
        # FSDD 官方划分：index 0-4 为 test，5-49 为 train
        is_test = idx < 5
        if is_test != (not train):
            n_split_skip += 1
            continue
        try:
            feat = log_mel_from_wav(
                f, ac["sample_rate"], n_mels, n_frames, ac["duration_sec"],
                norm_mode=norm_mode, norm_stats=norm_stats,
            )
            by_digit[digit].append(feat)
            paths_by_digit[digit].append(f)
        except Exception as e:
            if len(mel_errors) < 3:
                mel_errors.append(f"{os.path.basename(f)}: {e}")
            continue

    empty = [d for d, v in by_digit.items() if not v]
    if empty:
        hint = (
            f"共扫描 {len(files)} 个 wav；解析跳过 {n_parse_skip}，"
            f"划分跳过 {n_split_skip}，成功 {sum(len(v) for v in by_digit.values())}。"
        )
        if mel_errors:
            hint += " log-mel 失败示例: " + "; ".join(mel_errors)
            hint += "。请执行: pip install soundfile"
        raise RuntimeError(
            f"FSDD 在 {'train' if train else 'test'} 划分下缺少数字 {empty} 的 wav。"
            f"目录: {rec}。{hint}"
        )
    return by_digit, paths_by_digit


def _shift_feature(x, amount, dim):
    """无环绕地平移 2D 特征；空出的时频区域填 0。"""
    amount = int(amount)
    if amount == 0:
        return x
    size = x.size(dim)
    if abs(amount) >= size:
        return torch.zeros_like(x)
    out = torch.zeros_like(x)
    src = [slice(None), slice(None)]
    dst = [slice(None), slice(None)]
    if amount > 0:
        src[dim] = slice(0, size - amount)
        dst[dim] = slice(amount, size)
    else:
        src[dim] = slice(-amount, size)
        dst[dim] = slice(0, size + amount)
    out[tuple(dst)] = x[tuple(src)]
    return out


def _deterministic_audio_augment(feature, seed, augment_cfg):
    """为一个 pair_id 生成永久不变的轻量 log-mel 实例增广。"""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    out = feature.clone().float()

    max_time = max(0, int(augment_cfg.get("max_time_shift", 4)))
    max_freq = max(0, int(augment_cfg.get("max_freq_shift", 2)))
    if max_time:
        shift = int(torch.randint(
            -max_time, max_time + 1, (1,), generator=g).item())
        out = _shift_feature(out, shift, dim=1)
    if max_freq:
        shift = int(torch.randint(
            -max_freq, max_freq + 1, (1,), generator=g).item())
        out = _shift_feature(out, shift, dim=0)

    gain_min = float(augment_cfg.get("gain_min", 0.90))
    gain_max = float(augment_cfg.get("gain_max", 1.10))
    if gain_max < gain_min:
        raise ValueError("pairing.augment gain_max must be >= gain_min")
    gain = gain_min + (gain_max - gain_min) * torch.rand((), generator=g).item()
    out = out * gain

    noise_std = max(0.0, float(augment_cfg.get("noise_std", 0.01)))
    if noise_std:
        noise = torch.randn(out.shape, generator=g, dtype=out.dtype)
        out = out + noise_std * noise
    return out.clamp(0.0, 1.0)


class PairedAudioVisualDataset(Dataset):
    def __init__(self, cfg, train=True):
        data_cfg = cfg["data"]
        ac = cfg["audio"]
        self.num_classes = cfg["dims"]["num_classes"]
        self.n_mels, self.n_frames = audio_feature_shape(cfg)
        self.noise_std = ac["noise_std"]
        self.train = train
        self.pairing_cfg = data_cfg.get("pairing", {}) or {}
        self.fixed_augmented_pairing = bool(
            self.pairing_cfg.get("enabled", False)
            and self.pairing_cfg.get("mode", "") == "fixed_augmented_one_to_one")
        self.return_pair_id = bool(self.pairing_cfg.get("return_pair_id", False))
        self.pair_seed = int(self.pairing_cfg.get("seed", cfg.get("seed", 0)))
        self._split_salt = 0 if train else 1_000_000_007

        self.use_real_audio = bool(ac.get("use_real_audio", True))
        self._fsdd = None
        self._fsdd_paths = None
        if self.use_real_audio:
            self._fsdd, self._fsdd_paths = _load_fsdd_by_digit(cfg, train)
        self.toy_audio_prototype = not self.use_real_audio

        self.audio_protos = _make_audio_prototypes(
            self.num_classes, self.n_mels, self.n_frames)

        self._base = None
        if data_cfg.get("use_mnist", True):
            try:
                from torchvision import datasets, transforms
                tfm = transforms.ToTensor()
                self._base = datasets.MNIST(root=data_cfg["root"], train=train,
                                            download=True, transform=tfm)
                self._mode = "mnist"
            except Exception as e:
                print(f"[dataset] MNIST 不可用 ({e})，改用合成图像。", flush=True)
        if self._base is None:
            n = 6000 if train else 1000
            self._base = _SyntheticImages(n, self.num_classes,
                                          seed=0 if train else 1)
            self._mode = "synthetic"

        subset = data_cfg.get("train_subset", 0)
        if train and subset and subset > 0:
            self._indices = list(range(min(subset, len(self._base))))
        else:
            self._indices = list(range(len(self._base)))

        self._rng = np.random.default_rng(0 if train else 1)
        # 类别代表原型（class medoid），由 build_prototypes() 懒构建；
        # test 集通常复用 train 集原型（见 build_loaders）。
        self.prototype_img = None     # [C, 1, 28, 28]，真实 MNIST 样本
        self.prototype_aud = None     # [C, n_mels, n_frames]，真实 log-mel
        src = "FSDD+log-mel" if self.use_real_audio else "toy"
        print(f"[dataset] {'train' if train else 'test'} | 图像={self._mode} "
              f"n={len(self)} | 音频={src} shape=[{self.n_mels},{self.n_frames}]",
              flush=True)
        if self.fixed_augmented_pairing:
            print(
                "[dataset] 固定一一配对已启用：每张图像对应一个确定性增广音频 "
                f"pair_seed={self.pair_seed} return_pair_id={self.return_pair_id}",
                flush=True)

    def __len__(self):
        return len(self._indices)

    # ------------------------------------------------------------------
    # 类别代表原型（class medoid）：每类选一张距类中心最近的真实样本。
    # prototype_img[c] = argmin_i || image_i - mean(images_c) ||_2
    # prototype_aud[c] = argmin_i || logmel_i - mean(logmels_c) ||_2
    # 均为真实样本，不是 mean image / label 随机向量。
    # ------------------------------------------------------------------
    def build_prototypes(self):
        if self.prototype_img is None:
            self.prototype_img = self._build_image_prototypes()
        if self.prototype_aud is None:
            self.prototype_aud = self._build_audio_prototypes()
        print(f"[dataset] 已构建 class medoid 原型："
              f"image={tuple(self.prototype_img.shape)} "
              f"audio={tuple(self.prototype_aud.shape)}", flush=True)
        return self.prototype_img, self.prototype_aud

    @staticmethod
    def _medoid(stacked):
        """stacked: [n, ...] -> 距均值最近的那条样本 [...]（真实样本，非均值）。"""
        center = stacked.mean(dim=0, keepdim=True)
        dist = (stacked - center).flatten(1).pow(2).sum(dim=1)
        return stacked[int(dist.argmin())]

    def _build_image_prototypes(self):
        C = self.num_classes
        protos = torch.zeros(C, 1, 28, 28)
        base = self._base
        # 快路径：torchvision MNIST 暴露 .data(uint8)/.targets（与 ToTensor 一致 /255）
        if (self._mode == "mnist" and hasattr(base, "data")
                and hasattr(base, "targets")):
            data = base.data.float() / 255.0          # [N,28,28]
            targets = torch.as_tensor(base.targets)
            idx = torch.as_tensor(self._indices, dtype=torch.long)
            data = data[idx]
            targets = targets[idx]
            for c in range(C):
                mask = targets == c
                if int(mask.sum()) == 0:
                    continue
                protos[c, 0] = self._medoid(data[mask])
            return protos
        # 慢路径（合成图像或无 .data）：逐样本收集
        by_label = {c: [] for c in range(C)}
        for idx in self._indices:
            img, label = base[idx]
            by_label[int(label)].append(img.view(1, 28, 28))
        for c in range(C):
            if by_label[c]:
                protos[c] = self._medoid(torch.stack(by_label[c], dim=0))
        return protos

    def _build_audio_prototypes(self):
        C = self.num_classes
        protos = torch.zeros(C, self.n_mels, self.n_frames)
        if self.use_real_audio and self._fsdd is not None:
            for c in range(C):
                pool = self._fsdd.get(c, [])
                if not pool:
                    raise RuntimeError(f"FSDD 缺少数字 {c} 的音频样本，无法构建类别原型。")
                protos[c] = self._medoid(torch.stack(pool, dim=0))
            return protos
        # use_real_audio=false：仅冒烟，随机伪原型
        for c in range(C):
            protos[c] = self.audio_protos[c]
        return protos

    def _pair_id(self, item_index):
        """训练/测试命名空间不重叠的稳定整数 pair_id。"""
        return int(item_index) + (0 if self.train else 1_000_000_000)

    def _pair_spec(self, label, item_index, pool_size):
        """返回稳定的 (base_audio_index, augmentation_seed)。"""
        token = int(item_index) + self._split_salt
        base_index = (
            token * 104729 + int(label) * 1009 + self.pair_seed * 9176
        ) % int(pool_size)
        aug_seed = (
            self.pair_seed * 2_000_003
            + token * 1_000_033
            + int(label) * 10_007
        ) % (2 ** 63 - 1)
        return int(base_index), int(aug_seed)

    def _make_audio(self, label, item_index):
        if self.use_real_audio:
            pool = self._fsdd[label]
            if self.fixed_augmented_pairing:
                j, aug_seed = self._pair_spec(label, item_index, len(pool))
                return _deterministic_audio_augment(
                    pool[j], aug_seed, self.pairing_cfg.get("augment", {}))
            if self.train:
                j = int(self._rng.integers(0, len(pool)))
            else:
                # Evaluation pairing is a pure function of the dataset item.
                j = (int(item_index) * 104729 + int(label) * 1009) % len(pool)
            return pool[j].clone()
        proto = self.audio_protos[label]
        if self.fixed_augmented_pairing:
            _, aug_seed = self._pair_spec(label, item_index, 1)
            return _deterministic_audio_augment(
                proto, aug_seed, self.pairing_cfg.get("augment", {}))
        if self.train:
            noise = torch.randn(self.n_mels, self.n_frames)
        else:
            generator = torch.Generator().manual_seed(
                2_000_003 + int(item_index))
            noise = torch.randn(
                self.n_mels, self.n_frames, generator=generator)
        return (proto + self.noise_std * noise).clamp(0, 1)

    def __getitem__(self, idx):
        i = self._indices[idx]
        img, label = self._base[i]
        label = int(label)
        aud = self._make_audio(label, i)
        if self.return_pair_id:
            return img.float(), aud.float(), label, self._pair_id(i)
        return img.float(), aud.float(), label

    def _label_for_item(self, item_index):
        if hasattr(self._base, "targets"):
            return int(self._base.targets[int(item_index)])
        if hasattr(self._base, "labels"):
            return int(self._base.labels[int(item_index)])
        return int(self._base[int(item_index)][1])

    def export_pair_manifest(self, path):
        """导出可审计的一一配对清单，不实际复制扩增音频文件。"""
        if not self.fixed_augmented_pairing:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        split = "train" if self.train else "test"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "pair_id", "split", "image_index", "label",
                "base_audio", "base_audio_index", "augmentation_seed",
            ])
            for item_index in self._indices:
                label = self._label_for_item(item_index)
                pool_size = (len(self._fsdd[label])
                             if self.use_real_audio else 1)
                base_index, aug_seed = self._pair_spec(
                    label, item_index, pool_size)
                base_audio = "toy_prototype"
                if self.use_real_audio:
                    base_audio = os.path.basename(
                        self._fsdd_paths[label][base_index])
                writer.writerow([
                    self._pair_id(item_index), split, item_index, label,
                    base_audio, base_index, aug_seed,
                ])


def _seed_worker(_worker_id):
    """Give every training worker an independent, reproducible RNG stream."""
    from torch.utils.data import get_worker_info

    info = get_worker_info()
    if info is None:
        return
    worker_seed = int(info.seed % (2 ** 32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    info.dataset._rng = np.random.default_rng(worker_seed)
    base = getattr(info.dataset, "_base", None)
    if hasattr(base, "_rng"):
        base._rng = np.random.default_rng(worker_seed + 1)


def build_loaders(cfg):
    from torch.utils.data import DataLoader
    if cfg["audio"].get("use_real_audio", True):
        cfg["_audio_norm_stats"] = ensure_audio_norm_stats(cfg)
    else:
        cfg["_audio_norm_stats"] = None
    train_set = PairedAudioVisualDataset(cfg, train=True)
    test_set = PairedAudioVisualDataset(cfg, train=False)

    # 类别代表原型仅由 train 集构建，test/demo 复用同一份记忆原型，
    # 保证「类别代表」定义在训练数据上，评估/推理一致。
    train_set.build_prototypes()
    test_set.prototype_img = train_set.prototype_img
    test_set.prototype_aud = train_set.prototype_aud

    pairing_cfg = cfg.get("data", {}).get("pairing", {}) or {}
    manifest_dir = pairing_cfg.get("manifest_dir", "")
    if pairing_cfg.get("enabled", False) and manifest_dir:
        manifest_dir = str(resolve_from_root(manifest_dir))
        train_set.export_pair_manifest(os.path.join(
            manifest_dir, "pair_manifest_train.csv"))
        test_set.export_pair_manifest(os.path.join(
            manifest_dir, "pair_manifest_test.csv"))

    bs = cfg["data"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    train_generator = torch.Generator().manual_seed(int(cfg.get("seed", 0)))
    test_generator = torch.Generator().manual_seed(int(cfg.get("seed", 0)) + 1)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                              num_workers=nw, drop_last=True,
                              worker_init_fn=_seed_worker,
                              generator=train_generator)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False,
                             num_workers=nw, worker_init_fn=_seed_worker,
                             generator=test_generator)
    return train_loader, test_loader
