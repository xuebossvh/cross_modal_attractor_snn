"""Inventory AudioMNIST recordings and reuse the v16 speaker split."""

from pathlib import Path

from data.splits import audio_split, paper_split
from paths import resolve_from_root


def recordings(cfg):
    root = resolve_from_root(cfg["audio"]["audiomnist_root"])
    if (root / "data").is_dir():
        root = root / "data"
    expected = {f"{number:02}" for number in range(1, 61)}
    if not root.is_dir():
        raise FileNotFoundError(f"AudioMNIST data not found at {root}; set --audiomnist-root")
    pc = paper_split(cfg)
    validation = set(pc.get("val_speakers", []))
    test = set(pc.get("test_speakers", []))
    if (not pc.get("enabled") or pc.get("mode") != "speaker" or
            len(validation) != 9 or len(test) != 9 or
            validation & test or not (validation | test) <= expected):
        raise ValueError("AudioMNIST requires a disjoint 42/9/9 speaker split")
    by_split = {"train": [], "val": [], "test": []}
    for speaker in sorted(expected):
        for digit in range(10):
            for index in range(50):
                path = Path(root) / speaker / f"{digit}_{speaker}_{index}.wav"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing AudioMNIST recording: {path}")
                by_split[audio_split(path, cfg)].append(str(path))
    return by_split
