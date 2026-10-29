"""Deterministic, real-forward v15pro reconstruction panels.

The first dataset entries are shown without selecting successful reconstructions.
Truth support is displayed for evaluation, and enters forward only in oracle mode.
Use a separate output_dir for each mask source; existing exports are overwritten.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import default_collate

from common import unpack_paired_batch
from scripts.noise_metrics import (_conditions, deterministic_cues,
                                  preserve_random_state)


_COLUMNS = ("Clean target", "Noisy cue", "True support\n(evaluation only)",
            "Predicted support", "Applied repair mask", "Recovered")


def _checked(value, target, name, binary=False):
    if value is None:
        raise ValueError(f"Model did not return {name}")
    if value.shape != target.shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} != {tuple(target.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"Non-finite {name}")
    if binary and not torch.all((value == 0) | (value == 1)):
        raise ValueError(f"{name} must be binary")
    return value.detach().cpu()


def _plane(value):
    # Image tensors have a singleton channel; audio tensors may omit it.
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"Expected one 2-D grayscale/feature plane, got {tuple(value.shape)}")
    return value.numpy()


def _save_page(path, columns, modality, family, mode, mask_source, indices, ids, labels):
    count = len(indices)
    figure, axes = plt.subplots(count, 6, figsize=(15, 2.5 * count + 0.65),
                                squeeze=False)
    try:
        for row, sample_index in enumerate(indices):
            for column, values in enumerate(columns):
                axis = axes[row, column]
                cmap = "gray" if modality == "img" or column in (2, 3, 4) else "magma"
                axis.imshow(_plane(values[sample_index]), cmap=cmap, vmin=0, vmax=1,
                            origin="lower" if modality == "aud" else "upper",
                            interpolation="nearest", aspect="auto")
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(_COLUMNS[column], fontsize=10)
                if column == 0:
                    axis.set_ylabel(f"index={sample_index}\nid={ids[sample_index]}\n"
                                    f"label={int(labels[sample_index])}", fontsize=8)
        figure.suptitle(f"{modality} | {family} | {mode} | mask-source={mask_source}",
                        fontsize=12)
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        figure.savefig(path, dpi=140, metadata={"Description":
                       "First dataset entries; true support is evaluation-only except oracle."})
    finally:
        plt.close(figure)


@torch.no_grad()
def export_examples(model, loader, cfg, device, output_dir, split="test",
                    mask_source="predicted", num_examples=8) -> list[Path]:
    """Return generated PNG paths and tables/examples_manifest.json.

    num_examples=0 returns [] without creating files. Selection uses the first N
    dataset indices independently of the caller's sampler and batch size. Runtime
    model mode, mask-source setting and Python/NumPy/Torch RNG states are restored,
    including if validation or plotting raises an exception.
    """
    if isinstance(num_examples, bool) or not isinstance(num_examples, int) or num_examples < 0:
        raise ValueError("num_examples must be a non-negative integer")
    if num_examples == 0:
        return []
    if split not in ("val", "test"):
        raise ValueError("Example split must be val or test")
    if mask_source not in ("predicted", "oracle", "all"):
        raise ValueError("mask_source must be predicted, oracle, or all")
    if not cfg.get("noise_detector", {}).get("enabled", False):
        raise ValueError("Example export requires noise_detector.enabled")
    dataset = getattr(loader, "dataset", None)
    if dataset is None or not hasattr(dataset, "__getitem__"):
        raise ValueError("Example export requires an indexed evaluation dataset")
    if getattr(dataset, "train", False):
        raise ValueError("Examples cannot use the training dataset")
    actual_split = getattr(dataset, "split", None)
    if actual_split is not None and actual_split != split:
        raise ValueError(f"Requested {split} examples but loader contains {actual_split}")
    count = min(num_examples, len(dataset))
    if not count:
        raise ValueError("Empty evaluation dataset")
    settings = cfg.get("noise_evaluation", {}) or {}
    seed = int(settings.get("seed", 20261029))
    severity = float(settings.get("severity", cfg["corruption"]["train_severity"]))
    if not math.isfinite(severity) or not 0 <= severity <= 1:
        raise ValueError("Evaluation severity must be finite and in [0,1]")
    conditions = _conditions(cfg)
    root = Path(output_dir)
    figure_dir, table_dir = root / "figures", root / "tables"
    paths, manifest_rows = [], []
    original_source = model.noise_mask_source
    original_modes = [(module, module.training) for module in model.modules()]
    try:
        model.noise_mask_source = mask_source
        model.eval()
        with preserve_random_state():
            collate = getattr(loader, "collate_fn", None) or default_collate
            batch = collate([dataset[index] for index in range(count)])
            img, aud, labels, pair_ids = unpack_paired_batch(batch)
            img, aud, labels = img.cpu(), aud.cpu(), labels.cpu()
            ids = pair_ids.tolist() if pair_ids is not None else list(range(count))
            if len(ids) != count or len(labels) != count:
                raise ValueError("Collated examples do not match the selected dataset indices")
            figure_dir.mkdir(parents=True, exist_ok=True)
            table_dir.mkdir(parents=True, exist_ok=True)
            for family, mode in conditions:
                ci, ca, truth = deterministic_cues(
                    img, aud, ids, mode, family, cfg, severity, seed)
                # Normalize absent clean support for display and explicit oracle input.
                truth = {name: torch.zeros_like(target) if truth.get(name) is None
                         else truth[name] for name, target in (("img", img), ("aud", aud))}
                forward_masks = ({name: value.to(device) for name, value in truth.items()}
                                 if mask_source == "oracle" else {"img": None, "aud": None})
                out = model(x_img_cue=ci.to(device), x_aud_cue=ca.to(device),
                            training_mode=False, img_cue_mask=forward_masks["img"],
                            aud_cue_mask=forward_masks["aud"])
                files_by_index = [{} for _ in range(count)]
                for modality, target, cue in (("img", img, ci), ("aud", aud, ca)):
                    raw = _checked(out.get(f"recovered_{modality}"), target,
                                   f"recovered_{modality}")
                    recovered = raw.sigmoid() if modality == "img" else raw
                    predicted = _checked(out.get(f"{modality}_predicted_mask"), target,
                                         f"{modality}_predicted_mask", binary=True)
                    applied = _checked(out.get(f"{modality}_repair_mask"), target,
                                       f"{modality}_repair_mask", binary=True)
                    true_support = _checked(truth[modality], target, "true support", binary=True)
                    columns = (target, _checked(cue, target, "cue"), true_support,
                               predicted, applied, recovered)
                    for start in range(0, count, 4):
                        indices = list(range(start, min(start + 4, count)))
                        page = start // 4 + 1
                        path = figure_dir / f"examples_{modality}_{family}_{mode}_{page:02d}.png"
                        _save_page(path, columns, modality, family, mode, mask_source,
                                   indices, ids, labels)
                        paths.append(path)
                        for index in indices:
                            files_by_index[index][modality] = path.relative_to(root).as_posix()
                for index in range(count):
                    manifest_rows.append(dict(sample_index=index, sample_id=ids[index],
                                              label=int(labels[index]), seed=seed, split=split,
                                              family=family, cue_mode=mode, mask_source=mask_source,
                                              severity=0.0 if family == "clean" else severity,
                                              page=index // 4 + 1, row_on_page=index % 4,
                                              filenames=files_by_index[index]))
            manifest = dict(schema_version=1, selection="first N dataset indices; no quality filtering",
                            num_examples=count, requested_examples=num_examples, split=split,
                            seed=seed, mask_source=mask_source,
                            threshold=float(cfg["noise_detector"].get("threshold", 0.5)),
                            columns=list(_COLUMNS), display_range=[0, 1],
                            truth_support_usage="evaluation only; passed to model only for oracle",
                            examples=manifest_rows)
            path = table_dir / "examples_manifest.json"
            path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            paths.append(path)
    finally:
        model.noise_mask_source = original_source
        model.train(original_modes[0][1])
        for module, training in original_modes:
            module.training = training
    return paths
