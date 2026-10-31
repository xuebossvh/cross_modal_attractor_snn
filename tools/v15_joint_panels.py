"""Audited image/audio plates from one paired sample, without model inference.

The six columns expose the target, input, evaluation support, detector decision,
applied repair decision and result.  One plate is a representative observation,
not a population estimate.  Both modalities use the same fixed display range.
The NPZ and JSON are sufficient to redraw a plate without another forward pass.
"""

from __future__ import annotations

import importlib
import json
import math
import shutil
import sys
import tempfile
import textwrap
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import numpy as np


FIELDS = ("target", "cue", "true_mask", "predicted_mask", "repair_mask", "recovered")
MODALITIES = ("img", "aud")
SUFFIXES = (".png", ".pdf", ".npz", ".json", ".alignment.json", ".qa.json", ".svg")
_HEADINGS = ("Clean target", "Input cue", "True mask\n(evaluation only)",
             "Predicted mask", "Repair mask", "Recovered")
_REQUIRED_METADATA = ("sample_id", "index", "label", "index_prediction",
                      "training_seed", "policy", "family", "cue_mode",
                      "severity", "noise_std")


def joint_artifact_paths(output_stem) -> list[Path]:
    """Return the complete bundle paths; ``output_stem`` has no extension."""
    stem = Path(output_stem)
    return [stem.with_name(stem.name + suffix) for suffix in SUFFIXES]


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Metadata contains a non-finite value; use None for an empty region")
    return value


def _planes(arrays) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for modality in MODALITIES:
        if modality not in arrays:
            raise ValueError(f"Missing modality: {modality}")
        result[modality] = {}
        for field in FIELDS:
            if field not in arrays[modality]:
                raise ValueError(f"Missing array: {modality}.{field}")
            value = arrays[modality][field]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            plane = np.asarray(value)
            # A singleton image channel is accepted; batches must be sliced by caller.
            if plane.ndim == 3 and plane.shape[0] == 1:
                plane = plane[0]
            if plane.ndim != 2 or 0 in plane.shape:
                raise ValueError(f"{modality}.{field} must be a non-empty 2-D plane")
            if plane.dtype.kind not in "buif" or not np.isfinite(plane).all():
                raise ValueError(f"{modality}.{field} must contain finite real numbers")
            if field.endswith("mask") and not np.isin(plane, [0, 1]).all():
                raise ValueError(f"{modality}.{field} must be binary (0 or 1)")
            if np.any(plane < 0) or np.any(plane > 1):
                raise ValueError(f"{modality}.{field} lies outside the fixed [0, 1] display range")
            if field != "target" and plane.shape != result[modality]["target"].shape:
                raise ValueError(f"{modality}.{field} shape differs from its target")
            result[modality][field] = np.array(plane, copy=True)
    return result


def _foreground_coverage(metadata, modality):
    for container in ("coverage", "metrics"):
        entry = metadata.get(container, {}).get(modality, {})
        if isinstance(entry, Mapping) and "foreground_coverage" in entry:
            value = entry["foreground_coverage"]
            if value is None:
                return None
            value = float(value)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{modality}.foreground_coverage must be None or a fraction in [0, 1]")
            return value
    return None


def _metrics(planes, metadata):
    metrics = {}
    for modality, values in planes.items():
        support = values["true_mask"] == 1
        count = int(support.sum())
        target = values["target"].astype(np.float64)
        metrics[modality] = {
            "corrupted_pixels": count,
            "cue_corrupted_mse": (float(np.mean((values["cue"][support] - target[support]) ** 2))
                                  if count else None),
            "recovered_corrupted_mse": (float(np.mean((values["recovered"][support] - target[support]) ** 2))
                                        if count else None),
            "foreground_coverage": _foreground_coverage(metadata, modality),
        }
    return metrics


def _audit_modules(audit_scripts):
    """Use caller configuration, never a machine-specific path in this module."""
    added = None
    if audit_scripts is not None:
        directory = Path(audit_scripts).resolve()
        names = ("audit_panel_alignment", "audit_pdf_text", "audit_figure_collisions")
        if not all((directory / (name + ".py")).is_file() for name in names):
            raise RuntimeError("audit_scripts must contain all three nature-figure audit modules")
        added = str(directory)
        sys.path.insert(0, added)
    try:
        return tuple(importlib.import_module(name) for name in
                     ("audit_panel_alignment", "audit_pdf_text", "audit_figure_collisions"))
    except ImportError as exc:
        raise RuntimeError("Scientific-figure QA is required: pass audit_scripts or configure PYTHONPATH") from exc
    finally:
        if added is not None:
            sys.path.remove(added)


def _number(value):
    return "n/a" if value is None else f"{value:.4f}"


@contextmanager
def _output_lock(path):
    handle = path.open("x", encoding="utf-8")
    try:
        yield
    finally:
        handle.close()
        path.unlink()


def _draw(planes, metadata, metrics):
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    # Equal image-plate cells make spatial registration directly comparable.
    # This is an image plate + per-observation quantification, without uncertainty.
    title = (f"Pair {metadata['sample_id']}  |  label {metadata['label']}  |  "
             f"Index prediction {metadata['index_prediction']}  |  seed {metadata['training_seed']}  |  "
             f"split={metadata.get('split', 'unspecified')}")
    condition = (f"Noise: {metadata['family']}  |  location: {metadata['policy']}  |  "
                 f"cue: {metadata['cue_mode']}  |  severity {metadata['severity']}  |  "
                 f"noise SD {metadata['noise_std']}")
    title_lines = textwrap.wrap(title, width=108, break_long_words=False) or [title]
    condition_lines = textwrap.wrap(condition, width=122, break_long_words=False) or [condition]
    width_pt = 183 / 25.4 * 72
    height_pt = 304 + 11 * (len(title_lines) + len(condition_lines) - 2)
    style = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
             "font.size": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
             "axes.linewidth": 0.5, "text.usetex": False}
    with mpl.rc_context(style):
        figure = Figure(figsize=(width_pt / 72, height_pt / 72), dpi=300, facecolor="white")
        FigureCanvasAgg(figure)
        top = height_pt - 13
        for line in title_lines:
            figure.text(16 / width_pt, top / height_pt, line, fontsize=8, weight="bold", va="top")
            top -= 11
        top -= 2
        for line in condition_lines:
            figure.text(16 / width_pt, top / height_pt, line, fontsize=6.7, va="top")
            top -= 10
        grid = figure.add_gridspec(2, 6, left=16 / width_pt, right=1 - 8 / width_pt,
                                  bottom=42 / height_pt, top=226 / height_pt,
                                  hspace=32 / 76, wspace=8 / ((width_pt - 24 - 40) / 6))
        axes = [[figure.add_subplot(grid[row, col]) for col in range(6)] for row in range(2)]
        for row, modality in enumerate(MODALITIES):
            for col, field in enumerate(FIELDS):
                axis = axes[row][col]
                cmap = "gray" if modality == "img" or field.endswith("mask") else "magma"
                axis.imshow(planes[modality][field], cmap=cmap, vmin=0, vmax=1,
                            origin="lower" if modality == "aud" else "upper",
                            interpolation="nearest", aspect="auto")
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    # A fine frame makes an all-white repair mask visible on paper.
                    spine.set_visible(True)
                    spine.set_linewidth(0.35)
                    spine.set_edgecolor("#b0b0b0")
                if row == 0:
                    center = (axis.get_position().x0 + axis.get_position().x1) / 2
                    for line_index, line in enumerate(_HEADINGS[col].splitlines()):
                        figure.text(center, (258 - 9 * line_index) / height_pt, line,
                                    fontsize=7, ha="center", va="top")
            values = metrics[modality]
            coverage = values["foreground_coverage"]
            coverage_text = "n/a" if coverage is None else f"{coverage:.1%}"
            row_name = "Image" if modality == "img" else "Audio"
            summary = (f"{row_name}    Corrupted-region MSE (cue / recovered): "
                       f"{_number(values['cue_corrupted_mse'])} / {_number(values['recovered_corrupted_mse'])}"
                       f"    Foreground coverage: {coverage_text}")
            figure.text(16 / width_pt, axes[row][0].get_position().y1 + 11 / height_pt,
                        summary, fontsize=6.7, va="top")
        figure.text(16 / width_pt, 26 / height_pt,
                    "Masks: black 0 = retain / clean; white 1 = repair / corrupted. True mask: evaluation only.",
                    fontsize=6.5, va="top")
        figure.text(16 / width_pt, 15 / height_pt,
                    "Fixed display range [0, 1]; grayscale image, magma audio. n/a = empty region or undefined coverage.",
                    fontsize=6.3, va="top")
        figure.canvas.draw()
    return figure, style


def render_joint_sample(output_stem, arrays, metadata, *, audit_scripts=None) -> list[Path]:
    """Write an audited seven-file bundle for one image/audio pair.

    ``arrays[img|aud]`` contains target, cue, true_mask, predicted_mask,
    repair_mask and recovered. Arrays must be 2-D (a leading singleton channel
    is accepted), finite and in [0, 1]; masks must be binary. Coverage fractions
    use ``metadata['coverage'][modality]['foreground_coverage']``. The optional
    ``metrics`` container supports the same field as a fallback. Empty region
    MSE and unavailable/undefined foreground coverage are displayed as n/a.

    Mandatory QA modules are resolved from ``audit_scripts`` or PYTHONPATH.
    Alignment, text-size and collision errors block publication. Existing files
    are never overwritten. A renderer failure does not publish a partial bundle.
    """
    planes = _planes(arrays)
    metadata = _jsonable(dict(metadata))
    missing = [name for name in _REQUIRED_METADATA if name not in metadata]
    if missing:
        raise ValueError(f"Missing metadata fields: {', '.join(missing)}")
    # Reject non-JSON objects before allocating an output directory or rendering.
    json.dumps(metadata, allow_nan=False)
    metrics = _metrics(planes, metadata)
    alignment_module, text_module, collision_module = _audit_modules(audit_scripts)
    paths = joint_artifact_paths(output_stem)
    occupied = [path for path in paths if path.exists()]
    if occupied:
        raise FileExistsError(f"Refusing to overwrite existing joint-panel artifact: {occupied[0]}")
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    lock_path = stem.with_name(stem.name + ".render.lock")
    published = []
    with _output_lock(lock_path):
        try:
            with tempfile.TemporaryDirectory(prefix=".joint-panel-", dir=stem.parent) as temporary:
                stage = joint_artifact_paths(Path(temporary) / stem.name)
                figure, style = _draw(planes, metadata, metrics)
                try:
                    alignment = alignment_module.require_matplotlib_panel_alignment(
                        figure, json_out=stage[4], tolerance_pt=1.5,
                        gutter_tolerance_pt=1.5, strict=True)
                    import matplotlib as mpl
                    with mpl.rc_context(style):
                        figure.savefig(stage[0], dpi=300)
                        figure.savefig(stage[1], metadata={"Title": "Paired image/audio reconstruction",
                                                           "Subject": "One sample; true mask is evaluation only"})
                        figure.savefig(stage[6])
                finally:
                    figure.clear()
                text_report = text_module.audit_pdf(stage[1].read_bytes(), minimum_pt=5.0)
                if not text_report["auditable"] or text_report["below_minimum_count"]:
                    raise RuntimeError(f"PDF text QA failed: {json.dumps(text_report)}")
                collision_report = collision_module.audit_pdf(stage[1])
                # No intentional overlays are needed in this plate. Warnings block.
                if collision_module.exit_code(collision_report, strict=True):
                    raise RuntimeError(f"PDF collision QA failed: {json.dumps(collision_report)}")
                collision_report["pdf"] = paths[1].name
                qa = {"schema_version": 1, "alignment_verdict": alignment["verdict"],
                      "pdf_text": text_report, "pdf_collisions": collision_report,
                      "visual_review": "Not automated; inspect every panel at final size",
                      "source_preflight_notes": {
                          "EXPORT-RASTER": "PNG is the requested review artifact; editable PDF and SVG are supplied; TIFF is not requested.",
                          "RASTER-DPI": "PNG and embedded PDF/SVG images use 300 dpi; source arrays are preserved without synthetic upsampling.",
                          "FINAL-WIDTH": "Final physical width is 183 mm; measured dimensions are in the alignment report.",
                          "UNCERTAINTY-ENCODING": "Each plate contains one paired observation. MSE is over pixels, not a seed/fold/split aggregate; an uncertainty band is inapplicable.",
                      }}
                stage[5].write_text(json.dumps(qa, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                np.savez_compressed(stage[2], **{f"{modality}_{field}": planes[modality][field]
                                                for modality in MODALITIES for field in FIELDS})
                metadata["render_metrics"] = metrics
                metadata["render_contract"] = {
                    "schema_version": 1, "renderer": "v15_joint_panels", "width_mm": 183,
                    "columns": list(FIELDS), "rows": list(MODALITIES), "display_range": [0, 1],
                    "image_cmap": "gray", "audio_cmap": "magma", "mask_cmap": "gray",
                    "image_origin": "upper", "audio_origin": "lower", "interpolation": "nearest",
                    "true_mask_usage": "evaluation only", "mask_values": {"0": "retain / clean", "1": "repair / corrupted"},
                    "corrupted_mse_definition": "mean squared error over true_mask == 1; None if support is empty",
                    "uncertainty": "none: one paired observation, not an aggregate",
                    "npz_keys": [f"{m}_{f}" for m in MODALITIES for f in FIELDS],
                    "artifacts": [path.name for path in paths],
                }
                stage[3].write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                    encoding="utf-8")
                for source, destination in zip(stage, paths):
                    with destination.open("xb") as output, source.open("rb") as data:
                        published.append(destination)
                        shutil.copyfileobj(data, output)
        except BaseException:
            for path in published:
                path.unlink(missing_ok=True)
            raise
    return paths


def main(argv=None):
    """Redraw saved numeric data in a plotting environment without PyTorch."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--metadata-json", required=True, type=Path)
    parser.add_argument("--output-stem", required=True, type=Path)
    parser.add_argument("--audit-scripts", type=Path)
    args = parser.parse_args(argv)
    with np.load(args.input_npz, allow_pickle=False) as archive:
        arrays = {modality: {field: archive[f"{modality}_{field}"] for field in FIELDS}
                  for modality in MODALITIES}
    metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))
    paths = render_joint_sample(args.output_stem, arrays, metadata, audit_scripts=args.audit_scripts)
    print(json.dumps({"artifacts": [str(path) for path in paths]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
