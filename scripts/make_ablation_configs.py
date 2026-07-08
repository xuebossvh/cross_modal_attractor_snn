"""Generate ephemeral ablation configs under outputs/ablations_<base>/configs/.

Keeps configs/ to one YAML per official version (v10a … v10d). Ablation variants
are small diffs from the base config and are written locally before training.

Usage:
    python scripts/make_ablation_configs.py --base configs/v10d.yaml
    python scripts/make_ablation_configs.py --base configs/v10d.yaml --variant v10d_ab_refoff
"""

import argparse
import copy
from pathlib import Path

import bootstrap  # noqa: F401

import yaml

from common import load_config
from paths import PROJECT_ROOT


V10C_CUE_MODES = {
    "p_corrupt_img_only": 0.10,
    "p_corrupt_aud_only": 0.20,
    "p_corrupt_both": 0.20,
    "p_clean_img_only": 0.10,
    "p_clean_aud_only": 0.20,
    "p_clean_both": 0.20,
}

V10D_ABLATIONS = {
    "v10d_ab_v10cratio": {
        "summary": (
            "v10d structure + refiner, but v10c cue_modes ratio "
            "(isolates cue-ratio change)"
        ),
        "updates": {"cue_modes": V10C_CUE_MODES},
    },
    "v10d_ab_refoff": {
        "summary": (
            "v10d cue ratio + gated_dilated decoder, refiner disabled "
            "(isolates refiner + paste-back path)"
        ),
        "updates": {"audio_refiner": {"enabled": False}},
    },
}


def _deep_update(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value


def _set_paths(cfg, tag):
    cfg.setdefault("train", {})
    cfg.setdefault("decoder_pretrain", {})
    cfg["train"]["output_version"] = tag
    cfg["train"]["ckpt_path"] = f"outputs/checkpoints/cross_modal_snn_{tag}.pt"
    cfg["decoder_pretrain"]["ckpt_path"] = (
        f"outputs/checkpoints/cross_modal_snn_{tag}_decoder_pretrain.pt"
    )


def _ablation_root(base_config):
    stem = Path(base_config).stem
    return PROJECT_ROOT / "outputs" / f"ablations_{stem}" / "configs"


def build_variants(base_config="configs/v10d.yaml", variants=None):
    base = load_config(base_config)
    variants = variants or V10D_ABLATIONS
    built = []
    for tag, spec in variants.items():
        cfg = copy.deepcopy(base)
        _deep_update(cfg, spec["updates"])
        _set_paths(cfg, tag)
        built.append((tag, spec["summary"], cfg))
    return built


def write_variants(base_config="configs/v10d.yaml", out_dir=None, variants=None):
    out_dir = Path(out_dir) if out_dir else _ablation_root(base_config)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for tag, _summary, cfg in build_variants(base_config, variants):
        path = out_dir / f"{tag}.yaml"
        path.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        written.append((tag, path))
    return written


def _print_train_commands(written, base_config):
    rel_base = Path(base_config).as_posix()
    print(f"Base config: {rel_base}")
    print(f"Generated {len(written)} ablation config(s):\n")
    for tag, path in written:
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        log = f"outputs/{tag}/logs/train_{tag}_70ep.log"
        print(f"# {tag}")
        print(f"python scripts/mkdir_outputs.py --config {rel}")
        print("nohup env PYTHONUNBUFFERED=1 "
              f"python -u scripts/train.py --config {rel} > {log} "
              "2>&1 < /dev/null &")
        print(f"tail -f {log}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/v10d.yaml",
                    help="Official version config to derive ablations from.")
    ap.add_argument("--variant", default="",
                    help="Generate only this variant tag (default: all for base).")
    ap.add_argument("--out-dir", default="",
                    help="Override output directory.")
    args = ap.parse_args()

    variants = V10D_ABLATIONS
    if args.variant:
        if args.variant not in variants:
            known = ", ".join(sorted(variants))
            raise SystemExit(f"Unknown variant {args.variant!r}; known: {known}")
        variants = {args.variant: variants[args.variant]}

    out_dir = args.out_dir or None
    written = write_variants(args.base, out_dir=out_dir, variants=variants)
    _print_train_commands(written, args.base)


if __name__ == "__main__":
    main()
