"""Generate temporary configs for the first three v10a ablations.

The official branch config remains configs/v10a.yaml. This script writes
experiment configs under outputs/ablations_v10a/configs/ so the repository does
not accumulate one YAML file per ablation.
"""

import copy
from pathlib import Path

import bootstrap  # noqa: F401

import yaml

from common import load_config
from paths import PROJECT_ROOT


ABLATION_ROOT = PROJECT_ROOT / "outputs" / "ablations_v10a" / "configs"


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
        f"outputs/checkpoints/cross_modal_snn_{tag}_decoder_pretrain.pt")


def _variant(base, tag, updates):
    cfg = copy.deepcopy(base)
    _deep_update(cfg, updates)
    _set_paths(cfg, tag)
    return cfg


def build_variants(base):
    # A: baseline detail concat/pretrain, but remove phased input.
    yield "v10a_ablate_A_simultaneous", _variant(base, "v10a_ablate_A_simultaneous", {
        "detail_conditioning": {
            "enabled": True,
            "detach": False,
            "detach_value_for_recon": False,
            "zero_missing": True,
            "fusion": "concat",
            "img_detail_dim": 128,
            "aud_detail_dim": 128,
        },
        "decoder_pretrain": {"enabled": True},
        "index": {"input_schedule": "simultaneous"},
    })

    # B: v9-style run with only detach=false added, no decoder pretrain.
    yield "v10a_ablate_B_detach_false_only", _variant(base, "v10a_ablate_B_detach_false_only", {
        "detail_conditioning": {
            "enabled": True,
            "detach": False,
            "detach_value_for_recon": False,
            "zero_missing": True,
            "fusion": "concat",
            "img_detail_dim": 128,
            "aud_detail_dim": 128,
        },
        "decoder_pretrain": {"enabled": False},
        "index": {"input_schedule": "phased_img_first"},
    })

    # C: v9-style run with decoder pretrain only; keep detach=true.
    yield "v10a_ablate_C_pretrain_only", _variant(base, "v10a_ablate_C_pretrain_only", {
        "detail_conditioning": {
            "enabled": True,
            "detach": True,
            "detach_value_for_recon": False,
            "zero_missing": True,
            "fusion": "concat",
            "img_detail_dim": 128,
            "aud_detail_dim": 128,
        },
        "decoder_pretrain": {"enabled": True},
        "index": {"input_schedule": "phased_img_first"},
    })


def write_variants(config="configs/v10a.yaml", out_dir=ABLATION_ROOT):
    base = load_config(config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for tag, cfg in build_variants(base):
        path = out_dir / f"{tag}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False,
                                       allow_unicode=True), encoding="utf-8")
        written.append((tag, path))
    return written


def main():
    written = write_variants()

    print("Generated v10a ablation configs:")
    for tag, path in written:
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        log = f"outputs/{tag}/logs/train_{tag}_50ep.log"
        print(f"\n# {tag}")
        print(f"python scripts/mkdir_outputs.py --config {rel}")
        print("nohup env PYTHONUNBUFFERED=1 "
              f"python -u scripts/train.py --config {rel} > {log} "
              "2>&1 < /dev/null &")
        print(f"tail -f {log}")


if __name__ == "__main__":
    main()
