"""Prepare genuine instance-paired GRID data for v11e.

Expected extracted layout (directory names may contain extra nesting):

    AUDIO_ROOT/**/s1/bbaf2n.wav
    FRAMES_ROOT/**/s1/bbaf2n/*.jpg

The six-character GRID utterance id encodes the spoken digit at position 5
(``z`` means zero).  The image target is a rank-pooled 28x28 mouth-motion
dynamic image derived from frames of the *same utterance* as the audio.

This script creates precomputed tensors and one strict CSV manifest.  It uses
speaker-disjoint train/val/test splits and never expands augmented views into
additional pair ids.
"""

import bootstrap  # noqa: F401

import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from data.audio_features import _load_wav_mono, normalize_feature_global


UTTERANCE_RE = re.compile(r"([blps][bgrw][abiw][a-z][z1-9][anps])",
                          re.IGNORECASE)
SPEAKER_RE = re.compile(r"s\d+", re.IGNORECASE)


def _natural_key(value):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(value))]


def _speaker_from_path(path):
    for part in reversed(path.parts):
        if SPEAKER_RE.fullmatch(part):
            return part.lower()
    return None


def _utterance_from_text(text):
    match = UTTERANCE_RE.search(str(text))
    return match.group(1).lower() if match else None


def _digit_label(utterance):
    code = utterance[4].lower()
    return 0 if code == "z" else int(code)


def _index_frames(frames_root):
    index = {}
    extensions = {".jpg", ".jpeg", ".png"}
    for path in frames_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        speaker = _speaker_from_path(path)
        utterance = (_utterance_from_text(path.parent.name)
                     or _utterance_from_text(path.stem))
        if speaker and utterance:
            index.setdefault((speaker, utterance), []).append(path)
    for frames in index.values():
        frames.sort(key=_natural_key)
    return index


def _discover_records(audio_root, frames_root):
    frame_index = _index_frames(frames_root)
    records = []
    missing_frames = []
    for wav_path in sorted(audio_root.rglob("*.wav"), key=_natural_key):
        speaker = _speaker_from_path(wav_path)
        utterance = _utterance_from_text(wav_path.stem)
        if not speaker or not utterance:
            continue
        frames = frame_index.get((speaker, utterance), [])
        if not frames:
            missing_frames.append(str(wav_path))
            continue
        records.append({
            "speaker_id": speaker,
            "utterance": utterance,
            "source_id": f"{speaker}/{utterance}",
            "label": _digit_label(utterance),
            "wav_path": wav_path,
            "frames": frames,
        })
    if missing_frames:
        preview = "\n".join(missing_frames[:5])
        raise FileNotFoundError(
            f"{len(missing_frames)} GRID wav files have no matching frame "
            f"sequence. First entries:\n{preview}")
    if not records:
        raise FileNotFoundError(
            f"no matched GRID wav/frame pairs under {audio_root} and {frames_root}")
    sources = [row["source_id"] for row in records]
    if len(sources) != len(set(sources)):
        raise RuntimeError("duplicate GRID source ids were discovered")
    return records


def _speaker_splits(records, seed, val_speakers, test_speakers):
    speakers = sorted({row["speaker_id"] for row in records}, key=_natural_key)
    if len(speakers) <= val_speakers + test_speakers:
        raise ValueError(
            f"need more than {val_speakers + test_speakers} speakers, "
            f"found {len(speakers)}")
    shuffled = list(speakers)
    random.Random(seed).shuffle(shuffled)
    test = set(shuffled[:test_speakers])
    val = set(shuffled[test_speakers:test_speakers + val_speakers])
    train = set(shuffled[test_speakers + val_speakers:])
    split_by_speaker = {speaker: "train" for speaker in train}
    split_by_speaker.update({speaker: "val" for speaker in val})
    split_by_speaker.update({speaker: "test" for speaker in test})
    return split_by_speaker


def _rank_pool_motion(frame_paths, max_frames, crop):
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Pillow is required: pip install pillow") from e
    count = min(max_frames, len(frame_paths))
    indices = np.linspace(0, len(frame_paths) - 1, count).round().astype(int)
    frames = []
    for index in indices:
        with Image.open(frame_paths[int(index)]) as image:
            image = image.convert("L")
            width, height = image.size
            x0, y0, x1, y1 = crop
            box = (int(width * x0), int(height * y0),
                   int(width * x1), int(height * y1))
            image = image.crop(box).resize((28, 28))
            frames.append(np.asarray(image, dtype=np.float32) / 255.0)
    stack = np.stack(frames)
    if len(stack) == 1:
        dynamic = stack[0]
    else:
        # Approximate rank pooling: later frames get positive weight and early
        # frames negative weight, preserving temporal direction in one image.
        weights = np.linspace(-1.0, 1.0, len(stack), dtype=np.float32)
        dynamic = (stack * weights[:, None, None]).sum(axis=0)
    lo, hi = np.percentile(dynamic, [1.0, 99.0])
    if hi <= lo:
        return torch.zeros(1, 28, 28)
    dynamic = np.clip((dynamic - lo) / (hi - lo), 0.0, 1.0)
    return torch.from_numpy(dynamic.copy()).unsqueeze(0).float()


def _make_logmel_transform(sample_rate, n_mels, n_frames, duration_sec):
    try:
        import torchaudio
    except (ImportError, OSError) as e:
        raise RuntimeError(
            "torchaudio must match torch before GRID preprocessing") from e
    target_len = int(sample_rate * duration_sec)
    hop = max(1, target_len // n_frames)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=512, hop_length=hop, n_mels=n_mels)

    def convert(path):
        wav, source_rate = _load_wav_mono(str(path))
        if source_rate != sample_rate:
            wav = torchaudio.functional.resample(wav, source_rate, sample_rate)
        if wav.size(1) < target_len:
            wav = torch.nn.functional.pad(wav, (0, target_len - wav.size(1)))
        else:
            wav = wav[:, :target_len]
        feature = torch.log1p(transform(wav)).squeeze(0)
        if feature.size(1) < n_frames:
            feature = torch.nn.functional.pad(
                feature, (0, n_frames - feature.size(1)))
        return feature[:, :n_frames].float()

    return convert


def _compute_norm_stats(records, convert_audio, values_per_file, seed):
    generator = torch.Generator().manual_seed(seed)
    sampled = []
    train_records = [row for row in records if row["split"] == "train"]
    for index, row in enumerate(train_records):
        feature = convert_audio(row["wav_path"]).flatten()
        take = min(values_per_file, feature.numel())
        positions = torch.randint(
            0, feature.numel(), (take,), generator=generator)
        sampled.append(feature[positions])
        if index % 500 == 0:
            print(f"[GRID stats] {index}/{len(train_records)}", flush=True)
    values = torch.cat(sampled)
    lo = torch.quantile(values, 0.01).item()
    hi = torch.quantile(values, 0.99).item()
    if hi <= lo:
        hi = lo + 1e-6
    return {
        "lo": lo, "hi": hi, "percentile_lo": 1.0,
        "percentile_hi": 99.0, "n_wavs": len(train_records),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--frames_root", required=True)
    ap.add_argument("--output_root", default="_data/grid_v11e")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--val_speakers", type=int, default=3)
    ap.add_argument("--test_speakers", type=int, default=3)
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--duration_sec", type=float, default=3.0)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--n_frames", type=int, default=64)
    ap.add_argument("--max_video_frames", type=int, default=25)
    ap.add_argument("--norm_values_per_file", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug only; 0 processes the full dataset")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    audio_root = Path(args.audio_root).expanduser().resolve()
    frames_root = Path(args.frames_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    manifest_path = output_root / "pairs.csv"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(
            f"{manifest_path} exists; pass --force to rebuild it")

    print("[GRID] indexing real audiovisual source pairs...", flush=True)
    records = _discover_records(audio_root, frames_root)
    if args.limit > 0:
        records = records[:args.limit]
    split_by_speaker = _speaker_splits(
        records, args.seed, args.val_speakers, args.test_speakers)
    for row in records:
        row["split"] = split_by_speaker[row["speaker_id"]]
    counts = {split: sum(row["split"] == split for row in records)
              for split in ("train", "val", "test")}
    for split in counts:
        labels = {row["label"] for row in records if row["split"] == split}
        if labels != set(range(10)):
            raise RuntimeError(
                f"split={split} does not cover all digits: {sorted(labels)}")
    print(f"[GRID] unique real pairs: {counts}", flush=True)

    convert_audio = _make_logmel_transform(
        args.sample_rate, args.n_mels, args.n_frames, args.duration_sec)
    stats = _compute_norm_stats(
        records, convert_audio, args.norm_values_per_file, args.seed)
    stats.update({
        "sample_rate": args.sample_rate,
        "duration_sec": args.duration_sec,
        "n_mels": args.n_mels,
        "n_frames": args.n_frames,
    })
    output_root.mkdir(parents=True, exist_ok=True)
    torch.save(stats, output_root / "audio_norm_stats.pt")

    fieldnames = [
        "pair_id", "source_id", "image_source_id", "audio_source_id",
        "speaker_id", "utterance_id", "split", "label",
        "image_path", "audio_path",
    ]
    manifest_rows = []
    crop = (0.20, 0.35, 0.80, 0.90)
    for pair_id, row in enumerate(sorted(
            records, key=lambda item: _natural_key(item["source_id"]))):
        relative = Path(row["speaker_id"]) / f"{row['utterance']}.pt"
        image_rel = Path("images") / relative
        audio_rel = Path("audio_features") / relative
        image_out = output_root / image_rel
        audio_out = output_root / audio_rel
        image_out.parent.mkdir(parents=True, exist_ok=True)
        audio_out.parent.mkdir(parents=True, exist_ok=True)
        image = _rank_pool_motion(
            row["frames"], args.max_video_frames, crop)
        raw_audio = convert_audio(row["wav_path"])
        audio = normalize_feature_global(raw_audio, stats["lo"], stats["hi"])
        torch.save(image, image_out)
        torch.save(audio, audio_out)
        manifest_rows.append({
            "pair_id": pair_id,
            "source_id": row["source_id"],
            "image_source_id": row["source_id"],
            "audio_source_id": row["source_id"],
            "speaker_id": row["speaker_id"],
            "utterance_id": row["utterance"],
            "split": row["split"],
            "label": row["label"],
            "image_path": image_rel.as_posix(),
            "audio_path": audio_rel.as_posix(),
        })
        if pair_id % 500 == 0:
            print(f"[GRID prepare] {pair_id}/{len(records)}", flush=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary = {
        "dataset": "GRID",
        "pair_definition": "same physical utterance audio and video frames",
        "image_representation": "28x28 rank-pooled mouth-motion dynamic image",
        "unique_pairs": counts,
        "speakers": {
            split: sorted([speaker for speaker, value in split_by_speaker.items()
                           if value == split], key=_natural_key)
            for split in ("train", "val", "test")
        },
        "mnist_exposure_match": {
            "train_samples_per_epoch": 60000,
            "unique_train_pairs": counts["train"],
            "average_exposures_per_pair": 60000 / max(counts["train"], 1),
        },
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[GRID] manifest -> {manifest_path}", flush=True)
    print(
        f"[GRID] MNIST step match: 60000 exposures/epoch from "
        f"{counts['train']} unique train pairs "
        f"({60000 / max(counts['train'], 1):.2f}x average)", flush=True)


if __name__ == "__main__":
    main()
