"""Package clean P4 train/validation diagnostics and the intact epoch-100 state.

Run on the original CPU host, without downloading or modifying source data::

  python tools/package_v15_p4_assets.py --root outputs/v15pro --seed 1234 \
      --output outputs/v15pro_p4_assets --threads 1 --per-class 6

Only the small diagnostic cohorts become float32 NPZ files. Full A/B training
uses the GPU host's original training data, verified with the portable hashes.
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
import os
import shutil

import numpy as np
import torch
import yaml

from common import load_config
from data.audio_features import _fsdd_train_wav_paths
from data.dataset import PairedAudioVisualDataset
from data.splits import audio_split, norm_fingerprint
from models.frozen_base import file_sha256
from paths import resolve_from_root
from tools.package_v15_eval_assets import (
    arrays_fingerprint, collect_dataset, json_bytes, prepare_config,
    runtime_fingerprint, state_dict_sha256, write_json,
)


def validate_training_checkpoint(state, expected_epoch=99):
    """Require the actual resumable checkpoint; an inference subset is invalid."""
    if not isinstance(state, dict):
        raise ValueError("Expected a complete training checkpoint dictionary")
    packaging = state.get("asset_packaging", {})
    if packaging.get("inference_only") or packaging.get("resume_supported") is False:
        raise ValueError("An inference-only package cannot initialize P4 training")
    required = {"model", "opt", "sched", "cfg", "epoch", "noise_training", "frozen_base"}
    if required - state.keys():
        raise ValueError(f"Incomplete training checkpoint: missing {sorted(required - state.keys())}")
    if state["epoch"] != expected_epoch:
        raise ValueError(f"P4 requires epoch={expected_epoch}, got {state['epoch']}")
    if not isinstance(state["model"], dict) or not state["model"]:
        raise ValueError("Training checkpoint contains no model weights")
    opt = state["opt"]
    if (not isinstance(opt, dict) or not opt.get("state") or not opt.get("param_groups")):
        raise ValueError("Training checkpoint must retain optimizer state and parameter groups")
    if not isinstance(state["sched"], dict) or not state["sched"]:
        raise ValueError("Training checkpoint must retain its scheduler state")
    if not isinstance(state["cfg"], dict) or not state["cfg"]:
        raise ValueError("Training checkpoint must retain resolved cfg")
    if state["noise_training"].get("smoke_only", True):
        raise ValueError("Smoke checkpoints are not formal P4 initial states")
    return dict(keys=sorted(state), epoch=state["epoch"], completed_epochs=state["epoch"] + 1,
                inference_only=False, resume_supported=True, copied_without_reserialization=True,
                optimizer_state_entries=len(opt["state"]),
                optimizer_groups=[{k: deepcopy(v) for k, v in group.items() if k != "params"}
                                  for group in opt["param_groups"]],
                optimizer_group_parameter_counts=[len(group["params"]) for group in opt["param_groups"]],
                scheduler=deepcopy(state["sched"]),
                rng_policy=dict(original_checkpoint_fields_preserved=True,
                                epoch_seeded=bool(state["cfg"].get("train", {}).get("epoch_seeded", False))),
                noise_training=deepcopy(state["noise_training"]))


def build_clean_dataset(cfg, split):
    if split not in ("train", "val"):
        raise ValueError("P4 assets accept train/val only; test access is prohibited")
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    # The dataset constructor requests download=True. Preflight means it is a
    # no-op, and missing MNIST aborts before that constructor can download.
    local = MNIST(root=cfg["data"]["root"], train=True, download=False)
    del local
    dataset = PairedAudioVisualDataset(cfg, train=split == "train", split=split)
    if (dataset._mode != "mnist" or not dataset.use_real_audio
            or dataset.fixed_augmented_pairing or dataset.train != (split == "train")):
        raise ValueError("P4 requires the original real category-paired dataset")
    if type(dataset._base.transform) is not ToTensor:
        raise ValueError("Unexpected image augmentation: original transform must be ToTensor")
    return dataset


def pairing_metadata(dataset, seed):
    """Freeze original category pairing without pretending train is evaluation.

    Train pairing is random in the original code. Predeclare a single traversal
    of the original split order with the original dataset RNG seed. This is a
    diagnostic cohort, not a reconstruction of a shuffled training epoch.
    """
    if dataset.split not in ("train", "val"):
        raise ValueError("P4 diagnostics prohibit test samples")
    rng = np.random.default_rng(seed)
    records = []
    for position, item in enumerate(dataset._indices):
        item = int(item)
        label = dataset._label_for_item(item)
        pool = dataset._fsdd_paths[label]
        j = (int(rng.integers(0, len(pool))) if dataset.split == "train"
             else (item * 104729 + label * 1009) % len(pool))
        name = Path(pool[j]).name
        records.append(dict(position=position, image_index=item, label=label,
                            audio_pool_index=j, image_id=f"mnist_{dataset.split}_{item}",
                            audio_id=name, speaker="_".join(name[:-4].split("_")[1:-1])))
    return records


def select_balanced(records, per_class=6, num_classes=10):
    """Prefer smallest original image indices with distinct mapped audio IDs."""
    if per_class < 1:
        raise ValueError("per_class must be positive")
    selected, classes = [], {}
    for label in range(num_classes):
        candidates = sorted((r for r in records if r["label"] == label),
                            key=lambda r: r["image_index"])
        if len(candidates) < per_class:
            raise ValueError(f"Class {label} has fewer than {per_class} eligible images")
        chosen, seen = [], set()
        for record in candidates:
            if record["audio_id"] not in seen:
                chosen.append(record)
                seen.add(record["audio_id"])
                if len(chosen) == per_class:
                    break
        if len(chosen) < per_class:
            positions = {r["position"] for r in chosen}
            chosen.extend(r for r in candidates if r["position"] not in positions)
            chosen = chosen[:per_class]
        selected.extend(chosen)
        distinct = len({r["audio_id"] for r in chosen})
        classes[str(label)] = dict(samples=len(chosen), unique_audio=distinct,
            audio_diversity_shortfall=max(0, per_class - distinct),
            shortfall_reason=(None if distinct == per_class else
                              "Original category pairing provides too few distinct audio identities"))
    return sorted(selected, key=lambda r: r["image_index"]), classes


class FixedCleanSubset:
    """Lossless adapter for the existing evaluation asset collector."""

    def __init__(self, dataset, records):
        self.source, self.records, self.split = dataset, records, dataset.split
        self._indices = [r["image_index"] for r in records]
        self._fsdd_paths = dataset._fsdd_paths

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image, label = self.source._base[record["image_index"]]
        audio = self.source._fsdd[int(label)][record["audio_pool_index"]]
        sample_id = (self.source._pair_id(record["image_index"])
                     if self.source.return_pair_id else record["position"])
        return image.float(), audio.clone().float(), int(label), sample_id

    def evaluation_identity(self, index):
        return {k: self.records[index][k]
                for k in ("image_id", "audio_id", "speaker", "label")}


def training_data_fingerprint(cfg):
    """Portable raw MNIST tensors + per-file WAV hashes, never MNIST floats."""
    from torchvision.datasets import MNIST
    base = MNIST(root=cfg["data"]["root"], train=True, download=False)
    if base.data.dtype != torch.uint8:
        raise ValueError("Expected original uint8 MNIST training storage")
    arrays = dict(images=base.data.numpy(), labels=np.asarray(base.targets, dtype=np.int64))
    result = dict(mnist_train_content_sha256=arrays_fingerprint(arrays),
                  mnist_train_shape=list(base.data.shape), mnist_train_dtype="uint8",
                  mnist_training_files={})
    raw = Path(base.raw_folder)
    for name in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"):
        path = raw / name
        if path.is_file():
            result["mnist_training_files"][name] = dict(path=str(path.resolve()), sha256=file_sha256(path))
    del arrays, base
    paths = _fsdd_train_wav_paths(cfg)
    if not paths:
        raise ValueError("No original training WAVs available for verification")
    result["training_wav_sha256"] = {Path(p).name: file_sha256(p) for p in paths}
    result["training_wav_fingerprint"] = norm_fingerprint(paths, cfg)
    result["content_sha256"] = hashlib.sha256(json_bytes({
        k: result[k] for k in ("mnist_train_content_sha256", "training_wav_sha256",
                              "training_wav_fingerprint")})).hexdigest()
    return result


def verify_mapped_training_data(cfg, manifest, norm_stats_path):
    """Validate GPU-host paths against packaged train bytes and original stats.

    The caller may remap data.root/audio.fsdd_root/audio.norm_stats_path only;
    this helper reads the supplied paths and never recomputes normalization.
    """
    prepared = deepcopy(cfg)
    prepared["audio"]["auto_download"] = False
    configured_norm = resolve_from_root(prepared["audio"]["norm_stats_path"]).resolve()
    if configured_norm != Path(norm_stats_path).resolve():
        raise ValueError("Mapped cfg norm_stats_path must point to the verified normalization file")
    for section, expected_key, allowed in (
            ("audio", "audio_preprocessing", {"fsdd_root", "norm_stats_path", "auto_download"}),
            ("data", "data_protocol", {"root", "batch_size", "num_workers", "eval_split"})):
        expected = {k: v for k, v in manifest[expected_key].items() if k not in allowed}
        current = {k: v for k, v in cfg[section].items() if k not in allowed}
        if current != expected:
            raise ValueError(f"Mapped {section} configuration changed beyond permitted path/runtime fields")
    stats = torch.load(norm_stats_path, map_location="cpu", weights_only=True)
    if stats != manifest["normalization"]["stats"]:
        raise ValueError("Mapped normalization differs from original checkpoint training statistics")
    if prepared.get("_audio_norm_stats", stats) != stats:
        raise ValueError("Mapped cfg normalization differs from original checkpoint")
    if (stats.get("n_mels"), stats.get("n_frames")) != (
            prepared["audio"]["n_mels"], prepared["audio"]["n_frames"]):
        raise ValueError("Mapped normalization shape differs from audio configuration")
    observed = training_data_fingerprint(prepared)
    if observed["training_wav_fingerprint"] != stats.get("train_fingerprint"):
        raise ValueError("Mapped training WAV/config fingerprint differs from normalization")
    if observed["content_sha256"] != manifest["training_data"]["content_sha256"]:
        raise ValueError("Mapped training data differs from packaged source fingerprints")
    return dict(verified=True, training_data_content_sha256=observed["content_sha256"],
                normalization_sha256=file_sha256(norm_stats_path),
                normalization_recomputed=False, test_data_accessed=False)


def package_assets(root, output, seed=1234, *, per_class=6, best_reference="inference"):
    if seed != 1234:
        raise ValueError("P4 predeclares seed 1234; do not select seeds based on results")
    if per_class < 1 or best_reference not in ("inference", "none"):
        raise ValueError("Invalid diagnostic sample count or best reference mode")
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Use an independent output directory outside the source suite")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing assets: {output}")
    source = root / f"seed_{seed}"
    config_path, checkpoint = source / "config.yaml", source / "recovery/last.pt"
    best_path = source / "recovery/best.pt"
    for path in (config_path, checkpoint, best_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    original_sha, config_sha = file_sha256(checkpoint), file_sha256(config_path)
    cfg = load_config(config_path)
    if int(cfg.get("seed", -1)) != seed or int(cfg["dims"]["num_classes"]) != 10:
        raise ValueError("P4 requires the original seed1234 ten-class configuration")
    print("[p4-assets] validating full epoch-100 checkpoint with CPU mmap", flush=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    schema = validate_training_checkpoint(state)
    if int(state["cfg"].get("seed", -1)) != seed or cfg.get("train") != state["cfg"].get("train"):
        raise ValueError("Original config and last checkpoint seed/training configuration disagree")
    prepared, normalization = prepare_config(cfg, state["cfg"])
    if not isinstance(normalization["stats"], dict):
        raise ValueError("P4 requires original global training normalization statistics")
    model_sha = state_dict_sha256(state["model"])
    output.mkdir(parents=True)
    seed_output = output / f"seed_{seed}"
    seed_output.mkdir()
    (output / "datasets").mkdir()
    packed_checkpoint = seed_output / "last.pt"
    shutil.copyfile(checkpoint, packed_checkpoint)
    if file_sha256(packed_checkpoint) != original_sha:
        raise RuntimeError("Full checkpoint byte copy failed verification")
    shutil.copyfile(config_path, seed_output / "config.yaml")
    (seed_output / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    torch.save(deepcopy(normalization["stats"]), seed_output / "norm_stats.pt")
    normalization.update(file=f"seed_{seed}/norm_stats.pt",
                         file_sha256=file_sha256(seed_output / "norm_stats.pt"),
                         recomputed=False, path_migration_supported=True)
    del state
    gc.collect()
    runtime = runtime_fingerprint()
    runtime["p4_packager_sha256"] = file_sha256(__file__)
    manifest = dict(schema="v15pro-p4-assets.v1", status="incomplete", seed=seed,
        created_utc=datetime.now(timezone.utc).isoformat(), runtime=runtime,
        source_last_checkpoint=dict(path=str(checkpoint), sha256=original_sha,
                                    bytes=checkpoint.stat().st_size, epoch=99),
        source_best_checkpoint=dict(path=str(best_path), sha256=file_sha256(best_path),
                                    role="original validation-selected reference only"),
        source_schema=schema, checkpoint=f"seed_{seed}/last.pt",
        checkpoint_sha256=original_sha, model_state_sha256=model_sha,
        config=f"seed_{seed}/config.yaml", config_sha256=config_sha,
        resolved_config=f"seed_{seed}/resolved_config.yaml", normalization=normalization,
        audio_preprocessing=deepcopy(cfg["audio"]), data_protocol=deepcopy(cfg["data"]),
        diagnostic_contract=dict(
            permitted_splits=["train", "val"], test_data_accessed=False,
            image_transform="torchvision.transforms.ToTensor, exact float32",
            original_pairing=deepcopy(cfg["data"].get("pairing", {})),
            original_audio_augmentation=False, clean_audio_augmentation=False,
            augmentation_action="none: real-audio category pairing has no augmentation",
            original_audio_noise_std=cfg["audio"].get("noise_std"),
            noise_std_applies_to="synthetic audio only, not these real FSDD inputs",
            original_training_corruption=deepcopy(cfg.get("corruption", {})),
            clean_inputs_have_training_corruption=False,
            train_pairing="one original RNG draw per original split position, seed=1234; frozen for diagnostics",
            train_pairing_reconstructs_original_epoch=False,
            val_pairing="(image_index * 104729 + label * 1009) % class WAV pool size",
            selection="each class: smallest original image indices with distinct mapped audio first, then smallest remaining indices",
            selection_uses_model_outputs=False, per_class=per_class,
            sample_id_namespace="split + original evaluator position (or original pair_id if configured)",
            full_training_float_tensors_materialized=False,
            full_ab_training_uses="GPU host original data verified against training_data fingerprints",
            audio_reconstruction="unique_audio[audio_indices]"), datasets={}, artifacts={})
    write_json(output / "manifest.json", manifest)
    if best_reference == "inference":
        best = torch.load(best_path, map_location="cpu", weights_only=True, mmap=True)
        prepare_config(cfg, best.get("cfg"))
        subset = {k: best[k] for k in ("model", "cfg", "epoch", "noise_training", "frozen_base") if k in best}
        subset["asset_packaging"] = dict(inference_only=True, resume_supported=False,
            original_checkpoint_sha256=manifest["source_best_checkpoint"]["sha256"],
            omitted_keys=sorted(set(best) - set(subset)))
        torch.save(subset, seed_output / "best_reference.pt")
        manifest["source_best_checkpoint"].update(file=f"seed_{seed}/best_reference.pt",
            file_sha256=file_sha256(seed_output / "best_reference.pt"), epoch=best.get("epoch"),
            inference_only=True, resume_supported=False)
        del subset, best
        gc.collect()
    else:
        manifest["source_best_checkpoint"].update(file=None, copied=False)
    manifest["training_data"] = training_data_fingerprint(prepared)
    memberships = {}
    for split in ("train", "val"):
        dataset = build_clean_dataset(prepared, split)
        for pool in dataset._fsdd_paths.values():
            if any(audio_split(path, prepared) != split for path in pool):
                raise ValueError(f"A source audio identity crossed the {split} split boundary")
        records = pairing_metadata(dataset, seed)
        selected, classes = select_balanced(records, per_class, int(cfg["dims"]["num_classes"]))
        arrays, record = collect_dataset(FixedCleanSubset(dataset, selected))
        record.update(total_split_samples=len(dataset), selection=manifest["diagnostic_contract"]["selection"],
                      classes=classes, identities=selected,
                      sample_ids=arrays["sample_ids"].tolist(), image_indices=arrays["image_indices"].tolist())
        destination = output / "datasets" / f"{split}{len(selected)}.npz"
        np.savez_compressed(destination, **arrays)
        record.update(file=destination.relative_to(output).as_posix(), file_sha256=file_sha256(destination))
        manifest["datasets"][split] = record
        memberships[split] = dict(images=set(dataset._indices),
            audio={Path(p).name for pool in dataset._fsdd_paths.values() for p in pool},
            selected_images=set(arrays["image_indices"].tolist()), selected_audio=set(arrays["audio_ids"].tolist()))
        print(f"[p4-assets] {split}: {len(selected)} clean pairs, {record['unique_audio']} distinct audio", flush=True)
        del arrays, records, selected, dataset
        gc.collect()
        write_json(output / "manifest.json", manifest)
    overlap = {key: sorted(memberships["train"][key] & memberships["val"][key]) for key in memberships["train"]}
    if any(overlap.values()):
        raise ValueError(f"Training/validation split leakage: {overlap}")
    manifest["split_validation"] = dict(train_val_disjoint=True, overlap=overlap,
        checked_entire_split_membership=True, checked_selected_membership=True,
        sample_ids_are_split_scoped=True, test_tensors_loaded=False)
    if (file_sha256(checkpoint) != original_sha or file_sha256(config_path) != config_sha
            or file_sha256(best_path) != manifest["source_best_checkpoint"]["sha256"]):
        raise RuntimeError("Original checkpoint or configuration changed while packaging")
    manifest["artifacts"] = {p.relative_to(output).as_posix(): file_sha256(p)
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--best-reference", choices=("inference", "none"), default="inference")
    args = parser.parse_args(argv)
    if args.threads < 1 or args.per_class < 1:
        parser.error("threads and per-class must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    os.chdir(PROJECT_ROOT)
    package_assets(args.root, args.output, args.seed, per_class=args.per_class,
                   best_reference=args.best_reference)


if __name__ == "__main__":
    main()
