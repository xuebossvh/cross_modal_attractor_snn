"""Audit a completed P4 suite and evaluate fixed train60/val60 recovery stages.

This tool never trains, selects a new checkpoint, accesses test tensors, or
marks the shared experimental plan complete. Completed evaluations are reused
only after exact provenance and artifact hash verification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import html
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import torch

from common import set_seed
from models.network import CrossModalSNN
from scripts.evaluate_v15pro import verify_checkpoint_config
from tools.package_v15_eval_assets import arrays_fingerprint, runtime_fingerprint, state_dict_sha256
from tools.train_v15_p4 import (ORIGINAL_RUNTIME_SHA256, atomic_json,
    config_identity, file_sha256, state_sha256, verify_resume, verify_source_checkpoint)
from tools.v15_foreground_protocol import preserve_random_state
from tools.v15_p4_diagnostics import (array_sha256, conditions, evaluate_cases,
    load_clean_npz, make_case_cache)

BRANCHES = ("keep_lr", "restart_lr")
SPLITS = ("train", "val")
STAGES = ("input", "coarse", "refined", "final")
REGIONS = ("global", "corrupt_foreground", "corrupt_background", "clean_foreground")


def read_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON value {value} in {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def child(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Artifact path escapes its root: {relative}")
    return path


def check_files(root, hashes):
    for relative, expected in hashes.items():
        path = child(root, relative)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Artifact hash mismatch: {path}")


def verify_branches(bundles, source_sha256, expected_val_samples, expected_train_samples):
    """Validate equal budgets, stream identity, original best rule, and full-val scope."""
    identities = {}
    for branch in BRANCHES:
        manifest, epochs, status = (bundles[branch][key] for key in ("manifest", "epochs", "status"))
        identity = identities[branch] = manifest["identity"]
        if (identity["branch"] != branch or identity["source_sha256"] != source_sha256
                or identity["source_epoch"] != 99 or identity["epochs"] != 30
                or identity.get("smoke_only") is not False or identity.get("max_batches") != 0):
            raise ValueError(f"{branch}: wrong source, budget, branch, or smoke identity")
        if identity["runtime_sha256"] != ORIGINAL_RUNTIME_SHA256:
            raise ValueError(f"{branch}: original runtime mismatch")
        if identity["config"]["data"]["batch_size"] != 128:
            raise ValueError("Full P4 training must preserve batch size 128")
        expected_batches = expected_train_samples // 128  # Original drop_last=True.
        groups = identity["initialization"]["parameter_groups"]
        initial_lrs = [1e-5 if branch == "keep_lr" else 5e-5 * group["original_lr_multiplier"]
                       for group in groups]
        if [group["initial_lr"] for group in groups] != initial_lrs:
            raise ValueError(f"{branch}: wrong initial group learning rates")
        if ([record["epoch"] for record in epochs] != list(range(100, 130))
                or status.get("complete") is not True or status.get("last_epoch") != 129
                or status.get("smoke_only") is not False):
            raise ValueError(f"{branch}: expected exactly 30 complete formal epochs (100..129)")
        best_score, best_epoch, updates, samples = math.inf, None, 0, 0
        for record in epochs:
            if record.get("smoke_only") is not False:
                raise ValueError(f"{branch}: smoke epoch cannot enter the formal report")
            report = record["validation_report"]
            if (report.get("split") != "val" or report.get("mask_source") != "predicted"
                    or report.get("n_unique_samples") != expected_val_samples
                    or len(report.get("conditions", [])) != 7):
                raise ValueError(f"{branch}: incomplete or wrong validation scope")
            score = float(record["validation"]["score"])
            if not math.isfinite(score):
                raise ValueError(f"{branch}: non-finite original validation score")
            is_best = score < best_score
            if is_best:
                best_score, best_epoch = score, record["epoch"]
            if (record["is_best"] != is_best or record["best_epoch"] != best_epoch
                    or record["best_validation_score"] != best_score):
                raise ValueError(f"{branch}: saved best selection differs from the original strict-minimum rule")
            updates += record["optimizer_updates"]
            samples += record["samples"]
            if record["batches"] != expected_batches or record["samples"] != expected_batches * 128:
                raise ValueError(f"{branch}: training epoch did not cover the original full budget")
            offset = record["epoch"] - 100
            for key, step in (("lr_groups", offset), ("next_lr_groups", offset + 1)):
                expected = [1e-5 + (start - 1e-5) * (1 + math.cos(math.pi * step / 30)) / 2
                            for start in initial_lrs]
                actual = record[key]
                if len(actual) != len(expected) or any(not math.isclose(a, e, rel_tol=1e-10, abs_tol=1e-14)
                                                       for a, e in zip(actual, expected)):
                    raise ValueError(f"{branch}: unexpected LR trajectory at epoch {record['epoch']}")
            if (record["total_branch_updates"] != updates or record["total_branch_samples"] != samples
                    or record["optimizer_updates"] != record["batches"] or record["batches"] <= 0):
                raise ValueError(f"{branch}: inconsistent optimizer/sample budget")
        if (status["best_epoch"] != best_epoch or status["best_validation_score"] != best_score
                or status["total_branch_updates"] != updates):
            raise ValueError(f"{branch}: status disagrees with completed epoch records")
    left, right = (identities[branch] for branch in BRANCHES)
    for key in ("source_sha256", "source_epoch", "config", "config_sha256", "runtime_sha256",
                "driver_sha256", "state_helper_sha256", "packages", "torch_cuda_build", "device",
                "threads", "cuda_memory_fraction", "seed", "epochs", "normalization", "split_sha256"):
        if key not in left or left[key] != right.get(key):
            raise ValueError(f"A/B contract differs beyond the learning-rate branch: {key}")
    for key in ("initial_model_sha256", "initial_optimizer_sha256", "initial_optimizer_moments_sha256",
                "source_scheduler_sha256", "source_optimizer_steps", "trainable_names", "frozen_names"):
        if key not in left["initialization"] or left["initialization"][key] != right["initialization"].get(key):
            raise ValueError(f"A/B initial model/optimizer/freezing mismatch: {key}")
    parameter_groups = [[{k: v for k, v in group.items() if k != "initial_lr"}
                         for group in identity["initialization"]["parameter_groups"]]
                        for identity in (left, right)]
    if parameter_groups[0] != parameter_groups[1]:
        raise ValueError("A/B original optimizer parameter groups differ")
    for a, b in zip(bundles["keep_lr"]["epochs"], bundles["restart_lr"]["epochs"]):
        for key in ("clean_batch_stream_sha256", "epoch_seed", "batches", "samples", "optimizer_updates",
                    "cue_mode_batches", "total_branch_updates", "total_branch_samples"):
            if key not in a or a[key] != b.get(key):
                raise ValueError(f"A/B training stream/budget differs at epoch {a['epoch']}: {key}")
        if not a["clean_batch_stream_sha256"]:
            raise ValueError("Empty clean-batch stream identity")
    return dict(passed=True, source_model_optimizer_equal=True, matched_epochs=30,
        actual_clean_batch_stream_equal=True, equal_optimizer_updates=True,
        original_training_samples=expected_train_samples, training_batches_per_epoch=expected_batches,
        full_validation_samples=expected_val_samples,
        full_validation_conditions=7, selection="original strict minimum validation_score")


def verify_tiny(folder, source_sha256, datasets, cases):
    manifest = read_json(folder / "manifest.json")
    identity = manifest["identity"]
    updates = int(identity["updates"])
    if (identity.get("schema") != "v15-p4-tiny.v1" or manifest.get("status") != "complete" or manifest.get("completed_updates") != updates
            or updates != 1000 or identity.get("eval_every") != 100 or identity.get("seed") != 1234
            or identity.get("base_learning_rate") != 5e-5 or identity.get("batch_size") != 60
            or identity.get("source_checkpoint_sha256") != source_sha256
            or identity.get("source_epoch_zero_based") != 99
            or identity.get("source_runtime_sha256") != ORIGINAL_RUNTIME_SHA256):
        raise ValueError("Tiny diagnostic is incomplete or uses a different source/protocol")
    for name, expected in identity["diagnostic_source_sha256"].items():
        if file_sha256(child(ROOT / "tools", name)) != expected:
            raise ValueError(f"Tiny diagnostic implementation changed: {name}")
    expected_cues = {f"cue_cache/{split}/{key}.{extension}" for split in SPLITS
                     for key in cases[split] for extension in ("npz", "json")}
    if set(manifest["cue_cache_files"]) != expected_cues:
        raise ValueError("Tiny cue cache manifest does not cover exactly the fixed cases")
    if not {"history.jsonl", "metrics.json"} <= set(manifest["completed_artifacts_sha256"]):
        raise ValueError("Tiny completion manifest is missing its history/metrics")
    check_files(folder, manifest["cue_cache_files"])
    check_files(folder, manifest["completed_artifacts_sha256"])
    if file_sha256(folder / "checkpoints/last.pt") != manifest["last_checkpoint_sha256"]:
        raise ValueError("Tiny final checkpoint hash mismatch")
    checkpoint = torch.load(folder / "checkpoints/last.pt", map_location="cpu", weights_only=True)
    if (checkpoint.get("identity") != identity or checkpoint.get("completed_updates") != updates
            or checkpoint.get("source_checkpoint_sha256") != source_sha256
            or checkpoint.get("p4_diagnostic_only") is not True):
        raise ValueError("Tiny checkpoint disagrees with completed manifest")
    del checkpoint
    for split in SPLITS:
        if identity["datasets"][split]["arrays_sha256"] != array_sha256(datasets[split]):
            raise ValueError(f"Tiny {split} arrays differ from report inputs")
        if identity["case_sha256"][split] != {k: c["sha256"] for k, c in cases[split].items()}:
            raise ValueError(f"Tiny {split} corruption cases differ from report inputs")
        if identity["datasets"][split]["sample_ids"] != datasets[split]["sample_ids"].tolist():
            raise ValueError(f"Tiny {split} sample IDs differ from report inputs")
    history = [json.loads(line) for line in (folder / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    if [row["update"] for row in history] != list(range(1, updates + 1)):
        raise ValueError("Tiny update history is incomplete, reordered, or duplicated")
    if len(identity["schedule"]) != updates:
        raise ValueError("Tiny fixed-cue schedule has the wrong number of updates")
    for row, condition in zip(history, identity["schedule"]):
        if (row["condition"] != condition or row["case_sha256"] != cases["train"][condition]["sha256"]
                or not math.isfinite(float(row["loss"]))):
            raise ValueError("Tiny update does not match its declared fixed-cue schedule")
    expected_updates = [0, *range(identity["eval_every"], updates + 1, identity["eval_every"])]
    if expected_updates[-1] != updates:
        expected_updates.append(updates)
    evaluations = []
    for update in expected_updates:
        relative = f"evaluations/step_{update:06d}.json"
        if relative not in manifest["completed_artifacts_sha256"]:
            raise ValueError("Tiny evaluation is not covered by its completed artifact manifest")
        value = read_json(folder / relative)
        if value["update"] != update or set(value["splits"]) != set(SPLITS):
            raise ValueError("Tiny evaluation update/split mismatch")
        for split in SPLITS:
            verify_case_results(value["splits"][split], cases[split], 60)
            if update in (0, updates):
                for key in cases[split]:
                    relative_sample = f"samples/step_{update:06d}/{split}/{key}.npz"
                    if relative_sample not in manifest["completed_artifacts_sha256"]:
                        raise ValueError("Tiny initial/final samples are absent from completed artifact coverage")
        evaluations.append(value)
    return dict(manifest=manifest, history=history, evaluations=evaluations)


def verify_case_results(results, cases, samples):
    if set(results) != set(cases):
        raise ValueError("Fixed-case evaluation coverage differs")
    for key, row in results.items():
        if row["case_sha256"] != cases[key]["sha256"] or row["n"] != samples:
            raise ValueError(f"Fixed-case identity/sample count differs: {key}")
        if (not math.isfinite(float(row["total_loss"]))
                or not 0 <= row["index_accuracy"] <= 1 or not row["audio_refiner_bypassed"]):
            raise ValueError(f"Invalid case diagnostics: {key}")
        for modality, expected in (("img", set(STAGES)), ("aud", {"input", "coarse", "final"})):
            if set(row["modalities"][modality]) != expected:
                raise ValueError(f"Unexpected recovery stages: {key}/{modality}")
            for regions in row["modalities"][modality].values():
                if not set(REGIONS) <= set(regions):
                    raise ValueError("Recovery metrics omit required fixed regions")
                for stats in regions.values():
                    pixels, sse, mse = stats["pixels"], stats["sse"], stats["mse"]
                    if not isinstance(pixels, int) or pixels < 0 or not math.isfinite(sse) or sse < 0:
                        raise ValueError("Invalid region denominator/SSE")
                    if (pixels == 0 and (mse is not None or sse != 0)) or (
                            pixels > 0 and (mse is None or not math.isclose(mse, sse / pixels, rel_tol=1e-10, abs_tol=1e-14))):
                        raise ValueError("Region MSE does not equal SSE / pixels")


def evaluate_cached(folder, model, arrays, cases, cfg, device, identity, seed):
    """Only complete, exactly matching stages may be skipped after interruption."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    identity_file, completion = folder / "identity.json", folder / "completion.json"
    if identity_file.is_file():
        if read_json(identity_file) != identity:
            raise ValueError(f"Cached evaluation provenance changed; use a new report directory: {folder}")
    else:
        if any(folder.iterdir()):
            raise ValueError(f"Unidentified existing evaluation cache: {folder}")
        atomic_json(identity_file, identity)
    if completion.is_file():
        saved = read_json(completion)
        if saved.get("identity") != identity or saved.get("status") != "complete":
            raise ValueError(f"Invalid evaluation completion record: {folder}")
        expected_files = {"metrics.json", *(f"samples/{key}.npz" for key in cases)}
        if set(saved["artifacts_sha256"]) != expected_files:
            raise ValueError("Completed evaluation manifest omits or adds case artifacts")
        check_files(folder, saved["artifacts_sha256"])
        results = read_json(folder / "metrics.json")
        verify_case_results(results, cases, len(arrays["labels"]))
        print(f"[P4 report] verified cache {folder.name}", flush=True)
        return results
    with preserve_random_state():
        set_seed(seed + 900001)  # Exactly the tiny diagnostic evaluation RNG policy.
        results = evaluate_cases(model, arrays, cases, cfg, device,
                                 sample_directory=folder / "samples")
    verify_case_results(results, cases, len(arrays["labels"]))
    expected = {folder / "samples" / (key + ".npz") for key in cases}
    if set((folder / "samples").glob("*.npz")) != expected:
        raise ValueError("Saved diagnostic samples do not cover exactly the fixed cases")
    atomic_json(folder / "metrics.json", results)
    artifacts = {p.relative_to(folder).as_posix(): file_sha256(p)
                 for p in sorted([folder / "metrics.json", *expected])}
    atomic_json(completion, dict(status="complete", identity=identity, artifacts_sha256=artifacts))
    return results


def stage_rows(evaluations):
    """Keep conditions/regions separate; no averaging of region MSE ratios."""
    output = []
    for checkpoint, splits in evaluations.items():
        for split, results in splits.items():
            baseline = evaluations["source_last"][split]
            for key, case in results.items():
                for modality, stages in case["modalities"].items():
                    for region in REGIONS:
                        values = {stage: stages[stage][region] for stage in stages}
                        ref = baseline[key]["modalities"][modality]["final"][region]["mse"]
                        final = values["final"]["mse"]
                        output.append(dict(checkpoint=checkpoint, split=split, diagnostic_pairs=case["n"],
                            condition=key, policy=case["policy"], family=case["family"], cue_mode=case["cue_mode"],
                            modality=modality, region=region, stages=values,
                            final_minus_source_last_mse=None if final is None or ref is None else final - ref,
                            index_accuracy=case["index_accuracy"]))
    return output


def _format(value):
    if value is None:
        return "—"
    return f"{value:.7g}" if isinstance(value, float) else html.escape(str(value))


def _table(headers, rows):
    return "<div class='table'><table><thead><tr>" + "".join(
        f"<th>{html.escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{_format(v)}</td>" for v in row) + "</tr>" for row in rows
    ) + "</tbody></table></div>"


def render_html(summary):
    size = summary["scope"]["full_validation_pairs"]
    parts = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>P4 训练与恢复阶段诊断</title>",
        "<style>body{font:15px/1.6 system-ui;margin:32px;max-width:1600px;color:#172536}h1,h2{line-height:1.3}"
        ".notice{padding:16px;background:#edf4fa;border-left:4px solid #4479a4}.table{overflow:auto;margin:16px 0}"
        "table{border-collapse:collapse;font-size:13px}th,td{padding:7px 10px;border:1px solid #dbe2e9;white-space:nowrap}"
        "th{background:#eef2f6}details{margin:18px 0}summary{cursor:pointer;font-weight:bold}</style>",
        "<h1>P4：训练预算、学习率与恢复阶段诊断</h1>",
        f"<p class='notice'>完成检查覆盖 A/B 各 30 轮和固定 train60 / val60 阶段评价。"
        f"完整验证集为 {size} 对；val60 是固定诊断子集。选模仍使用原七条件完整验证 score。"
        "本报告不读取 test，不改变选模规则，不自动将共享实验计划标为完成。</p>",
        "<p>图像阶段：coarse → refined（实际 delta 后裁剪、回填前）→ final。"
        "音频 refiner 保持旁路，仅有 coarse → final（回填）；表中的缺失阶段显示 —。"
        "Index accuracy 是内部类别读出准确率，不能作为恢复内容的独立语义识别率。</p>",
        "<p><a href='summary.json'>完整 machine-readable summary.json</a></p>",
        "<h2>A/B 完整验证轨迹</h2>"]
    for branch in BRANCHES:
        records = summary["full_validation"][branch]["epochs"]
        parts += [f"<h3>{html.escape(branch)}</h3>", _table(
            ["epoch", "train loss", f"full val ({size}) score", "image MSE", "audio MSE", "best epoch", "LR groups"],
            [[r["epoch"], r["epoch_loss"], r["validation"]["score"], r["validation"].get("img_mse"),
              r["validation"].get("aud_mse"), r["best_epoch"], json.dumps(r["lr_groups"])] for r in records])]
    parts += ["<h2>固定 60 对样本：恢复阶段与来源模型差值</h2>",
        "<p>每行保持 split、噪声类型、位置策略、cue 与区域独立。MSE 由该区域 SSE / 像素数计算；"
        "空区域为 —。Δ final 是该模型 final 减 source_last final，负值表示误差下降。"
        "没有跨条件平均，也没有将复用音频的配对数解释为独立音频数。所有原始 SSE、方向梯度误差、"
        "饱和比例和 Index 指标保存在 JSON 与评价缓存。</p>"]
    parts += ["<p>饱和比例包括合法背景零值，不等于裁剪失败率；真实裁剪越界数记录在 image_refiner_clip。"
              "方向梯度仅统计两端都在该区域内的相邻边，不代表跨污染边界误差。</p>"]
    for checkpoint in summary["diagnostics"]:
        rows = [r for r in summary["stage_rows"] if r["checkpoint"] == checkpoint]
        parts += [f"<details><summary>{html.escape(checkpoint)} · train60 / val60</summary>", _table(
            ["split", "policy", "family", "cue", "modality", "region", "pixels", "input MSE", "coarse MSE",
             "refined MSE", "final MSE", "Δ final vs source last", "final saturation"],
            [[r["split"], r["policy"], r["family"], r["cue_mode"], r["modality"], r["region"],
              r["stages"]["final"]["pixels"], *[r["stages"].get(s, {}).get("mse") for s in STAGES],
              r["final_minus_source_last_mse"], r["stages"]["final"]["saturation_ratio"]] for r in rows]), "</details>"]
    parts += ["<h2>固定小样本拟合过程</h2>",
        f"<p>固定 train60 共 {summary['tiny']['updates']} 次更新，使用新建 Adam。"
        "该拟合实验不与 A/B 保留动量的完整训练等同。中间损失按更新所用条件保留；"
        "同条件阶段评价的完整数值可从下表链接读取。</p>", _table(
            ["update", "固定阶段评价 JSON"],
            [[row["update"], row["path"]] for row in summary["tiny"]["evaluation_files"]]),
        "<p>本次为单 seed 诊断。A/B 仅检验末期学习率路径下继续训练的效果；"
        "不能仅凭这些结果判定模块容量不足、污染位置是唯一原因或直接替代 P5/P6 的独立实验。</p>",
        "</html>"]
    return "\n".join(parts)


def run(args):
    if args.threads < 1 or not math.isfinite(args.cuda_memory_fraction) or not 0 < args.cuda_memory_fraction <= 1:
        raise ValueError("threads must be positive and cuda-memory-fraction must be in (0,1]")
    root, assets = Path(args.root).resolve(), Path(args.assets).resolve()
    output = root / "report"
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA is unavailable; no CPU fallback")
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device=cuda_index)
    torch.set_num_threads(args.threads)
    asset_manifest = read_json(assets / "manifest.json")
    if (asset_manifest.get("schema") != "v15pro-p4-assets.v1"
            or asset_manifest.get("status") != "complete" or asset_manifest.get("seed") != 1234
            or asset_manifest.get("diagnostic_contract", {}).get("permitted_splits") != ["train", "val"]
            or asset_manifest.get("diagnostic_contract", {}).get("per_class") != 6
            or not asset_manifest.get("split_validation", {}).get("train_val_disjoint")):
        raise ValueError("Expected complete seed1234 P4 train60/val60 assets")
    check_files(assets, asset_manifest["artifacts"])
    runtime = runtime_fingerprint()
    if runtime["source_sha256"] != ORIGINAL_RUNTIME_SHA256:
        raise ValueError("Original runtime source hash changed")
    source_path = child(assets, asset_manifest["checkpoint"])
    source_sha = file_sha256(source_path)
    if source_sha != asset_manifest["checkpoint_sha256"]:
        raise ValueError("Source checkpoint differs from asset provenance")
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    cfg = verify_source_checkpoint(source)
    if (int(cfg["seed"]) != 1234 or cfg["_audio_norm_stats"] != asset_manifest["normalization"]["stats"]
            or cfg["audio"] != asset_manifest["audio_preprocessing"]
            or cfg["data"] != asset_manifest["data_protocol"]):
        raise ValueError("Original source config/data protocol differs from packaged provenance")
    if state_dict_sha256(source["model"]) != asset_manifest["model_state_sha256"]:
        raise ValueError("Source model hash differs from the asset manifest")
    datasets = {split: load_clean_npz(child(assets, asset_manifest["datasets"][split]["file"])) for split in SPLITS}
    tiny_manifest = read_json(root / "tiny/manifest.json")
    seed = int(tiny_manifest["identity"]["seed"])
    cases = {split: make_case_cache(arrays, cfg, seed) for split, arrays in datasets.items()}
    for split in SPLITS:
        if len(datasets[split]["labels"]) != 60:
            raise ValueError("P4 report requires 60 fixed clean pairs in each diagnostic split")
        metadata = asset_manifest["datasets"][split]
        if (metadata["split"] != split or metadata["samples"] != 60
                or metadata["content_sha256"] != arrays_fingerprint(datasets[split])
                or metadata["sample_ids"] != datasets[split]["sample_ids"].tolist()
                or metadata["image_indices"] != datasets[split]["image_indices"].tolist()):
            raise ValueError(f"Packaged {split} content or identity metadata mismatch")
        path = child(assets, metadata["file"])
        if (file_sha256(path) != metadata["file_sha256"]
                or tiny_manifest["identity"]["datasets"][split]["file_sha256"] != metadata["file_sha256"]):
            raise ValueError(f"Tiny {split} asset file differs from packaged provenance")
    if set(datasets["train"]["image_indices"].tolist()) & set(datasets["val"]["image_indices"].tolist()):
        raise ValueError("Fixed training and validation image memberships overlap")
    if set(datasets["train"]["audio_ids"].tolist()) & set(datasets["val"]["audio_ids"].tolist()):
        raise ValueError("Fixed training and validation audio memberships overlap")
    tiny = verify_tiny(root / "tiny", source_sha, datasets, cases)
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if tiny["manifest"]["identity"]["source_config_sha256"] != config_hash:
        raise ValueError("Tiny source config differs from original checkpoint")
    bundles = {branch: {name: read_json(root / branch / (name + ".json"))
                       for name in ("manifest", "epochs", "status")} for branch in BRANCHES}
    full_val = int(asset_manifest["datasets"]["val"]["total_split_samples"])
    audit = verify_branches(bundles, source_sha, full_val,
                            int(asset_manifest["datasets"]["train"]["total_split_samples"]))
    for branch in BRANCHES:
        identity = bundles[branch]["manifest"]["identity"]
        if (identity["config"] != config_identity(cfg)
                or identity["initialization"]["initial_model_sha256"] != state_dict_sha256(source["model"])
                or identity["initialization"]["initial_optimizer_sha256"] != state_sha256(source["opt"])):
            raise ValueError("A/B initial state/config disagrees with the packaged source checkpoint")
    best_asset = asset_manifest["source_best_checkpoint"]
    if not best_asset.get("file"):
        raise ValueError("Original validation-selected best_reference checkpoint is required")
    checkpoints = dict(source_last=source_path, source_best=child(assets, best_asset["file"]))
    for branch in BRANCHES:
        checkpoints.update({branch + "_" + name: root / branch / (name + ".pt") for name in ("best", "last")})
    input_paths = [assets / "manifest.json", root / "tiny/manifest.json", root / "tiny/history.jsonl",
        *[root / branch / (name + ".json") for branch in BRANCHES for name in ("manifest", "epochs", "status")],
        *checkpoints.values()]
    input_hashes = {str(path): file_sha256(path) for path in input_paths}
    diagnostic_sources = {name: file_sha256(ROOT / "tools" / name) for name in (
        "report_v15_p4.py", "v15_p4_diagnostics.py", "v15_foreground_protocol.py",
        "train_v15_p4.py", "package_v15_eval_assets.py")}
    common_identity = dict(schema="v15-p4-report.v1", source_sha256=source_sha,
        config_sha256=config_hash, input_files_sha256=input_hashes,
        runtime_sha256=runtime["source_sha256"], tool_sha256=diagnostic_sources,
        packages=runtime["packages"], device=str(device), threads=args.threads,
        cuda_memory_fraction=args.cuda_memory_fraction if device.type == "cuda" else None,
        seed=seed, test_data_accessed=False)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        if read_json(manifest_path)["identity"] != common_identity:
            raise ValueError("Report provenance changed; existing cache cannot be reused")
    elif any(output.iterdir()):
        raise ValueError("Existing report directory has no provenance manifest")
    atomic_json(manifest_path, dict(identity=common_identity, status="evaluating"))
    del source
    gc.collect()
    evaluations, references = {}, {}
    for stage, checkpoint in checkpoints.items():
        print(f"[P4 report] evaluating {stage} on fixed train60/val60", flush=True)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        verify_checkpoint_config(cfg, state.get("cfg"))
        if config_identity(cfg) != config_identity(state["cfg"]):
            raise ValueError(f"Checkpoint config drift: {stage}")
        if stage.startswith(BRANCHES):
            branch, suffix = stage.rsplit("_", 1)
            verify_resume(state, bundles[branch]["manifest"]["identity"])
            expected_epoch = 129 if suffix == "last" else bundles[branch]["status"]["best_epoch"]
            if state["epoch"] != expected_epoch or state["p4"]["records"] != bundles[branch]["epochs"][:expected_epoch - 99]:
                raise ValueError(f"Checkpoint and full validation records disagree: {stage}")
        elif stage == "source_last" and state.get("epoch") != 99:
            raise ValueError("Source last checkpoint is not epoch 99")
        elif stage == "source_best":
            if (file_sha256(checkpoint) != best_asset["file_sha256"]
                    or state.get("epoch") != best_asset["epoch"]
                    or state.get("asset_packaging", {}).get("original_checkpoint_sha256") != best_asset["sha256"]):
                raise ValueError("Original best reference lineage mismatch")
        model_sha = state_dict_sha256(state["model"])
        model = CrossModalSNN(cfg).to(device)
        model.load_state_dict(state["model"], strict=True)
        references[stage] = dict(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                                epoch=state["epoch"], model_sha256=model_sha)
        del state
        evaluations[stage] = {}
        for split in SPLITS:
            identity = dict(common_identity, stage=stage, split=split,
                checkpoint_sha256=references[stage]["checkpoint_sha256"], model_sha256=model_sha,
                arrays_sha256=array_sha256(datasets[split]),
                case_sha256={k: c["sha256"] for k, c in cases[split].items()})
            folder = output / "evaluation_cache" / stage / split
            evaluations[stage][split] = evaluate_cached(folder, model, datasets[split], cases[split],
                                                       cfg, device, identity, seed)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    evaluations["tiny_initial"] = tiny["evaluations"][0]["splits"]
    evaluations["tiny_final"] = tiny["evaluations"][-1]["splits"]
    if any(file_sha256(path) != digest for path, digest in input_hashes.items()):
        raise ValueError("An input artifact changed while the report was being generated")
    check_files(root / "tiny", tiny["manifest"]["completed_artifacts_sha256"])
    summary = dict(schema="v15-p4-report.v1", status="complete_awaiting_scientific_review",
        created_utc=datetime.now(timezone.utc).isoformat(), identity=common_identity, fairness_audit=audit,
        scope=dict(diagnostic_train_pairs=60, diagnostic_val_pairs=60,
            full_validation_pairs=full_val, diagnostic_subset_is_not_full_validation=True,
            test_data_accessed=False, training_seed=1234, index_accuracy_is_recovered_semantics=False,
            source_partitions=asset_manifest["split_validation"]),
        checkpoints=references, diagnostics=evaluations, stage_rows=stage_rows(evaluations),
        full_validation={branch: dict(epochs=bundles[branch]["epochs"], status=bundles[branch]["status"])
                         for branch in BRANCHES},
        tiny=dict(updates=tiny["manifest"]["identity"]["updates"], history=tiny["history"],
            evaluation_files=[dict(update=e["update"], path=f"../tiny/evaluations/step_{e['update']:06d}.json")
                              for e in tiny["evaluations"]]),
        interpretation=dict(image="coarse -> clipped refinement before paste-back -> final",
            audio="coarse -> final paste-back; audio refiner remains bypassed",
            region_aggregation="SSE / pixel count for each condition and region; empty regions are null",
            selection="original full-validation score; diagnostic subset does not select checkpoints",
            conclusions="Single-seed descriptive diagnostics; P5 and P6 remain separate future experiments"))
    atomic_json(output / "summary.json", summary)
    html_path = output / "report.html"
    temporary = output / "report.html.tmp"
    temporary.write_text(render_html(summary), encoding="utf-8")
    temporary.replace(html_path)
    atomic_json(manifest_path, dict(identity=common_identity, status="complete_awaiting_scientific_review",
        artifacts_sha256={name: file_sha256(output / name) for name in ("summary.json", "report.html")}))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cuda-memory-fraction", type=float, default=.6)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
