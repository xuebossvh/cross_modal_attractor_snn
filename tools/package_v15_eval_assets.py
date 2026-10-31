"""Package exact clean v15 evaluation inputs and inference-only checkpoints.

Run from the original project environment (no GPU, training, or downloads):
  python tools/package_v15_eval_assets.py --root outputs/v15pro \
      --output outputs/v15pro_eval_assets --seeds 1234 2345 3456

The NPZ schema stores images as float32 and deduplicates audio by source ID.
Recover a sample's audio with unique_audio[audio_indices[i]]. sample_ids retain
the original evaluator's IDs (sequential indices when return_pair_id is false).
Checkpoint subsets retain the original cfg and load with load_strict_checkpoint;
they omit optimizer/scheduler/RNG state and MUST NOT be used to resume training.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil

import numpy as np
import torch
import yaml

from common import load_config
from data.audio_features import _fsdd_train_wav_paths
from data.dataset import PairedAudioVisualDataset
from data.splits import norm_fingerprint
from models.frozen_base import file_sha256
from paths import resolve_from_root
from scripts.evaluate_v15pro import verify_checkpoint_config


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def runtime_fingerprint():
    """Use exactly export_v15pro.runtime_source_fingerprint's file selection."""
    files = [PROJECT_ROOT / "common.py", PROJECT_ROOT / "paths.py"]
    for directory in ("models", "data", "scripts"):
        files.extend((PROJECT_ROOT / directory).glob("*.py"))
    digest, hashes = hashlib.sha256(), {}
    for path in sorted(files):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        raw = path.read_bytes()
        digest.update(relative.encode())
        digest.update(raw)
        hashes[relative] = hashlib.sha256(raw).hexdigest()
    versions = {}
    for name in ("torch", "torchvision", "torchaudio", "numpy", "soundfile", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(source_sha256=digest.hexdigest(), source_files=hashes,
                python=sys.version, platform=platform.platform(), packages=versions,
                torch_version=str(torch.__version__), torch_cuda_build=torch.version.cuda,
                export_device="cpu", torch_threads=torch.get_num_threads(),
                packager_sha256=file_sha256(__file__))


def state_dict_sha256(model_state):
    """Canonical tensor-content digest, independent of torch ZIP serialization."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model_state.items()):
        if not torch.is_tensor(tensor) or tensor.layout != torch.strided:
            raise TypeError(f"Expected a dense tensor state_dict entry: {name}")
        value = tensor.detach().cpu().contiguous()
        digest.update(json_bytes([name, str(value.dtype), list(value.shape)]))
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def arrays_fingerprint(arrays, names=None):
    digest = hashlib.sha256()
    for name in sorted(arrays if names is None else names):
        array = np.ascontiguousarray(arrays[name])
        digest.update(json_bytes([name, array.dtype.str, list(array.shape)]))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def prepare_config(cfg, checkpoint_cfg):
    """Read the saved training normalization, never recompute or write it."""
    verify_checkpoint_config(cfg, checkpoint_cfg)
    if cfg.get("data", {}).get("dataset", "mnist_fsdd") != "mnist_fsdd":
        raise ValueError("This exact-input packager supports v15 MNIST/FSDD only")
    if not cfg["data"].get("paper_split", {}).get("enabled", False):
        raise ValueError("Independent publication train/val/test splits are required")
    if not cfg["data"].get("use_mnist", True) or not cfg["audio"].get("use_real_audio", True):
        raise ValueError("Synthetic fallbacks cannot be packaged as original v15 assets")
    pairing = cfg["data"].get("pairing", {}) or {}
    if pairing.get("enabled", False):
        raise ValueError("Expected unaugmented v15 category pairing; enabled pairing is unsupported")
    prepared = deepcopy(cfg)
    stats = checkpoint_cfg.get("_audio_norm_stats")
    if cfg["audio"].get("norm_mode", "global") != "per_sample" and not isinstance(stats, dict):
        raise ValueError("Original checkpoint must contain training-only _audio_norm_stats")
    if "_audio_norm_stats" in cfg and cfg["_audio_norm_stats"] != stats:
        raise ValueError("Generated config and checkpoint normalization differ")
    prepared["_audio_norm_stats"] = deepcopy(stats)
    prepared["audio"]["auto_download"] = False
    stats_path = resolve_from_root(cfg["audio"].get("norm_stats_path", "_data/audio_norm_stats.pt"))
    stats_file = None
    if stats_path.is_file() and stats is not None:
        saved_stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        if saved_stats != stats:
            raise ValueError(f"Normalization file differs from the training checkpoint: {stats_path}")
        stats_file = dict(path=str(stats_path), sha256=file_sha256(stats_path))
    if stats is not None:
        paths = _fsdd_train_wav_paths(prepared)
        if not paths or stats.get("train_fingerprint") != norm_fingerprint(paths, prepared):
            raise ValueError("Training WAV/config fingerprint differs from saved normalization")
        if (stats.get("n_mels"), stats.get("n_frames")) != (
                cfg["audio"]["n_mels"], cfg["audio"]["n_frames"]):
            raise ValueError("Saved training normalization shape differs from audio config")
    return prepared, dict(source="checkpoint.cfg._audio_norm_stats", stats=stats,
                          original_stats_file=stats_file,
                          training_wav_fingerprint_verified=stats is not None)


def build_eval_dataset(cfg, split):
    # build_loaders always builds train medoids for MNIST/FSDD, even with
    # train_required=False. The evaluation __getitem__ path never uses them.
    # Check local MNIST first: PairedAudioVisualDataset's download=True becomes
    # a no-op on these already-present files; missing data aborts without I/O.
    from torchvision.datasets import MNIST
    local = MNIST(root=cfg["data"]["root"], train=split != "test", download=False)
    del local
    dataset = PairedAudioVisualDataset(cfg, train=False, split=split)
    if dataset._mode != "mnist" or dataset.fixed_augmented_pairing or dataset.train:
        raise ValueError("Evaluation must use real MNIST and deterministic, unaugmented audio")
    return dataset


def collect_dataset(dataset, limit=0):
    """Materialize eval images only; each unaugmented WAV feature is stored once."""
    count = min(len(dataset), limit) if limit else len(dataset)
    if count < 1:
        raise ValueError("Empty evaluation dataset")
    images, unique_audio, audio_lookup = None, [], {}
    labels, sample_ids, image_indices, audio_indices = [], [], [], []
    identities, source_audio = [], {}
    for index in range(count):
        sample = dataset[index]
        image, audio, label = sample[:3]
        identity = dataset.evaluation_identity(index)
        item = int(dataset._indices[index])
        if int(label) != identity["label"]:
            raise ValueError("Dataset tensor and identity labels disagree")
        image = image.detach().cpu().contiguous().numpy()
        audio = audio.detach().cpu().contiguous().numpy()
        if image.dtype != np.float32 or audio.dtype != np.float32:
            raise ValueError("Expected original float32 evaluation tensors")
        if not np.isfinite(image).all() or not np.isfinite(audio).all():
            raise ValueError("Non-finite clean evaluation tensor")
        if images is None:
            images = np.empty((count, *image.shape), dtype=np.float32)
        images[index] = image
        audio_id = identity["audio_id"]
        if audio_id in audio_lookup:
            audio_index = audio_lookup[audio_id]
            if not np.array_equal(unique_audio[audio_index], audio):
                raise ValueError(f"Same WAV ID produced different tensors: {audio_id}")
        else:
            audio_index = len(unique_audio)
            audio_lookup[audio_id] = audio_index
            unique_audio.append(audio.copy())
        labels.append(int(label))
        sample_ids.append(int(sample[3]) if len(sample) > 3 else index)
        image_indices.append(item)
        audio_indices.append(audio_index)
        identities.append(identity)
        if index and index % 2000 == 0:
            print(f"[assets] {dataset.split}: {index}/{count}", flush=True)
    # Record bytes of every source WAV in the evaluation pool, including those
    # outside a first-256 validation selection. No training tensors are loaded.
    for pool in dataset._fsdd_paths.values():
        for path in pool:
            source_audio[Path(path).name] = file_sha256(path)
    arrays = dict(images=images, unique_audio=np.stack(unique_audio),
                  audio_indices=np.asarray(audio_indices, dtype=np.int64),
                  labels=np.asarray(labels, dtype=np.int64),
                  sample_ids=np.asarray(sample_ids, dtype=np.int64),
                  image_indices=np.asarray(image_indices, dtype=np.int64),
                  image_ids=np.asarray([v["image_id"] for v in identities]),
                  audio_ids=np.asarray([v["audio_id"] for v in identities]),
                  speakers=np.asarray([v["speaker"] for v in identities]))
    metadata_keys = ("labels", "sample_ids", "image_indices", "image_ids", "audio_ids", "speakers")
    record = dict(split=dataset.split, total_split_samples=len(dataset), samples=count,
                  selection="all" if count == len(dataset) else f"first_{count}_in_original_order",
                  unique_audio=len(unique_audio), source_audio_sha256=source_audio,
                  metadata_sha256=arrays_fingerprint(arrays, metadata_keys),
                  content_sha256=arrays_fingerprint(arrays),
                  arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype))
                          for key, value in arrays.items()})
    return arrays, record


def package_assets(root, output, seeds, *, val_samples=256, expected_test=10000,
                   expected_val=5999):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("Use a new output directory outside the original suite root")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing directory: {output}")
    jobs = []
    for seed in seeds:
        folder = root / f"seed_{seed}"
        config, checkpoint = folder / "config.yaml", folder / "recovery/best.pt"
        for path in (config, checkpoint):
            if not path.is_file():
                raise FileNotFoundError(path)
        jobs.append((seed, folder, config, checkpoint))
    output.mkdir(parents=True)
    (output / "shared").mkdir()
    runtime = runtime_fingerprint()
    manifest = dict(schema="v15pro-eval-assets.v1", status="incomplete",
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    original_suite_root=str(root), runtime=runtime,
                    evaluation_contract=dict(
                        image_transform="torchvision.transforms.ToTensor (float32)",
                        audio_pairing="(MNIST item index * 104729 + label * 1009) % class WAV pool size",
                        audio_augmentation=False, training_tensors_materialized=False,
                        image_order="original split order, no shuffle",
                        tensor_values="exact float32, lossless NPZ compression",
                        audio_reconstruction="unique_audio[audio_indices]",
                        sample_ids="dataset pair IDs if returned; otherwise sequential evaluator indices"),
                    seeds={}, artifacts={})
    write_json(output / "manifest.json", manifest)
    shared_records = {}
    for seed, folder, config, checkpoint in jobs:
        print(f"[assets] seed {seed}: loading checkpoint with CPU mmap", flush=True)
        config_sha, original_sha = file_sha256(config), file_sha256(checkpoint)
        cfg = load_config(config)
        if int(cfg.get("seed", seed)) != seed:
            raise ValueError(f"Config seed does not match seed_{seed}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(state, dict) or "model" not in state:
            raise ValueError(f"Expected a complete training checkpoint: {checkpoint}")
        prepared, normalization = prepare_config(cfg, state.get("cfg"))
        seed_output = output / f"seed_{seed}"
        seed_output.mkdir()
        shutil.copyfile(config, seed_output / "config.yaml")
        # The original YAML is byte-for-byte; resolved YAML remains usable even
        # when an input unexpectedly has an extends chain.
        (seed_output / "resolved_config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        model_sha = state_dict_sha256(state["model"])
        subset = {key: state[key] for key in
                  ("model", "cfg", "epoch", "noise_training", "frozen_base") if key in state}
        subset["asset_packaging"] = dict(inference_only=True, resume_supported=False,
            original_checkpoint_sha256=original_sha, model_state_sha256=model_sha,
            omitted_keys=sorted(set(state) - set(subset)))
        packed_checkpoint = seed_output / "best.pt"
        torch.save(subset, packed_checkpoint)
        if file_sha256(checkpoint) != original_sha or file_sha256(config) != config_sha:
            raise RuntimeError("Original checkpoint/config changed during packaging")
        record = dict(seed=seed, config=f"seed_{seed}/config.yaml", config_sha256=config_sha,
                      resolved_config=f"seed_{seed}/resolved_config.yaml",
                      checkpoint=f"seed_{seed}/best.pt", checkpoint_epoch=state.get("epoch"),
                      original_checkpoint=str(checkpoint), original_checkpoint_sha256=original_sha,
                      checkpoint_sha256=file_sha256(packed_checkpoint),
                      model_state_sha256=model_sha,
                      inference_only=True, resume_supported=False,
                      normalization=normalization, audio_preprocessing=deepcopy(cfg["audio"]),
                      preprocessing_details=dict(n_fft=512, hop_length=max(1, int(
                          cfg["audio"]["sample_rate"] * cfg["audio"]["duration_sec"]) //
                          cfg["audio"]["n_frames"]), log_transform="torch.log1p",
                          mel_defaults="torchaudio.transforms.MelSpectrogram defaults from source runtime"),
                      data_protocol=deepcopy(cfg["data"]), datasets={})
        original_manifest = folder / "inference/tables/export_manifest.json"
        if original_manifest.is_file():
            identity = json.loads(original_manifest.read_text(encoding="utf-8")).get("identity", {})
            if identity.get("checkpoint_sha256") != original_sha:
                raise ValueError("Original inference manifest checkpoint differs from best.pt")
            record["original_inference_identity"] = identity
            record["runtime_matches_original_inference"] = (
                identity.get("runtime_sha256") == runtime["source_sha256"])
        del subset, state
        gc.collect()
        for split, limit, expected in (("test", 0, expected_test), ("val", val_samples, expected_val)):
            dataset = build_eval_dataset(prepared, split)
            if expected and len(dataset) != expected:
                raise ValueError(f"Expected {expected} {split} samples, got {len(dataset)}")
            arrays, dataset_record = collect_dataset(dataset, limit)
            previous = shared_records.get(split)
            if previous is not None and previous["content_sha256"] == dataset_record["content_sha256"]:
                dataset_record["file"] = previous["file"]
                dataset_record["file_sha256"] = previous["file_sha256"]
                dataset_record["shared_with_seed"] = seeds[0]
            else:
                name = "test.npz" if split == "test" else f"val_first_{val_samples}.npz"
                destination = (output / "shared" if previous is None else seed_output / "datasets") / name
                destination.parent.mkdir(exist_ok=True)
                np.savez_compressed(destination, **arrays)
                dataset_record["file"] = destination.relative_to(output).as_posix()
                dataset_record["file_sha256"] = file_sha256(destination)
                if previous is None:
                    shared_records[split] = deepcopy(dataset_record)
            record["datasets"][split] = dataset_record
            print(f"[assets] seed {seed} {split}: {dataset_record['samples']} pairs, "
                  f"{dataset_record['unique_audio']} WAV features, "
                  f"SHA256={dataset_record['content_sha256']}", flush=True)
            del arrays, dataset
            gc.collect()
        manifest["seeds"][str(seed)] = record
        write_json(output / "manifest.json", manifest)
    manifest["cross_seed_identical"] = {
        split: len({record["datasets"][split]["content_sha256"]
                    for record in manifest["seeds"].values()}) == 1
        for split in ("test", "val")}
    manifest["artifacts"] = {p.relative_to(output).as_posix(): file_sha256(p)
                             for p in sorted(output.rglob("*"))
                             if p.is_file() and p.name != "manifest.json"}
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "manifest.json", manifest)
    print(f"[assets] complete: {output}; cross-seed identical: "
          f"{manifest['cross_seed_identical']}", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 2345, 3456])
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--expected-test", type=int, default=10000)
    parser.add_argument("--expected-val", type=int, default=5999)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or min(args.seeds) < 0 or args.val_samples < 1 or args.threads < 1
            or min(args.expected_test, args.expected_val) < 0):
        parser.error("Unique nonnegative seeds and positive samples/threads are required")
    # Keep CPU use bounded on the original 0.5-core/2-GB host.
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    os.chdir(PROJECT_ROOT)
    package_assets(args.root, args.output, args.seeds, val_samples=args.val_samples,
                   expected_test=args.expected_test, expected_val=args.expected_val)


if __name__ == "__main__":
    main()
