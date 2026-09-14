"""Profile completed v13pro checkpoints without changing training or test metrics.

The operation counts are deliberately labelled estimates: dense MACs are an upper
bound and active MACs scale them by observed nonzero input fraction. They are not
hardware energy or exact event-driven SOP measurements.
"""

import bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path
import time

import torch
from torch import nn

from common import build_cue, load_config, unpack_paired_batch
from data.dataset import build_loaders
from models.network import CrossModalSNN
from models.paper_baselines import CleanRecognizers
from paths import resolve_from_root
from scripts.paper_baseline import make_model


NON_SNN = ("classifier", "recognizer", "cue_cnn", "conditioned_cnn", "matched_cnn")


def _tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return next((_tensor(x) for x in value if _tensor(x) is not None), None)
    return None


class OpCounter:
    def __init__(self):
        self.rows = []
        self.handles = []

    def attach(self, model):
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            self.handles.append(module.register_forward_hook(self._hook(name, module)))

    def _hook(self, name, module):
        def hook(_, inputs, output):
            x = _tensor(inputs)
            y = _tensor(output)
            if x is None or y is None:
                return
            if isinstance(module, nn.Linear):
                out_features = module.out_features
                in_features = module.in_features
                batch_ops = x.numel() // in_features
                dense = batch_ops * in_features * out_features
            else:
                _, out_channels, out_h, out_w = y.shape
                k_h, k_w = module.kernel_size
                dense = (y.shape[0] * out_channels * out_h * out_w *
                         (module.in_channels // module.groups) * k_h * k_w)
            fraction = float((x.abs() > 1e-8).float().mean())
            self.rows.append(dict(module=name, type=module.__class__.__name__,
                                  dense_macs=int(dense), active_macs=float(dense * fraction),
                                  input_nonzero_fraction=fraction))
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _forward(model, kind, cfg, batch, device, protos):
    img, aud, labels, _ = unpack_paired_batch(batch)
    img, aud = img.to(device), aud.to(device)
    ci, ca, masks = build_cue(img, aud, "corrupt_both", cfg, severity=.4,
                              img_mode="occlusion", aud_mode="time_mask", return_masks=True)
    if kind in NON_SNN:
        if kind == "recognizer":
            out = (model.img(img), model.aud(aud))
        elif kind == "classifier":
            out = (model.img(ci) + model.aud(ca)) / 2
        else:
            out = model(ci, ca, masks, *protos)
    else:
        out = model(x_img_cue=ci, x_aud_cue=ca, img_cue_mask=masks["img"],
                    aud_cue_mask=masks["aud"], training_mode=False)
    return out, len(labels)


def profile_one(config_path, repeats=8):
    cfg = load_config(config_path)
    kind = cfg.get("paper", {}).get("experiment", "unknown")
    ckpt_path = resolve_from_root(cfg["train"]["eval_ckpt_path"])
    record = dict(config=str(config_path), experiment=kind, checkpoint=str(ckpt_path))
    if not ckpt_path.is_file():
        record["status"] = "missing_checkpoint"
        return record
    device = torch.device("cuda" if cfg.get("device") == "cuda" and torch.cuda.is_available() else "cpu")
    _, loader = build_loaders(cfg, eval_split="test")
    batch = next(iter(loader))
    dataset = loader.dataset
    protos = (dataset.prototype_img.to(device), dataset.prototype_aud.to(device))
    state = torch.load(ckpt_path, map_location="cpu")
    model = (CleanRecognizers() if kind == "recognizer" else
             (CrossModalSNN(cfg) if kind not in NON_SNN else make_model(kind, cfg))).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    counter = OpCounter()
    counter.attach(model)
    with torch.no_grad():
        for _ in range(2):
            _forward(model, kind, cfg, batch, device, protos)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        out, batch_size = None, 0
        for _ in range(max(1, repeats)):
            out, batch_size = _forward(model, kind, cfg, batch, device, protos)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    counter.close()
    dense = sum(row["dense_macs"] for row in counter.rows) / max(1, repeats)
    active = sum(row["active_macs"] for row in counter.rows) / max(1, repeats)
    output_rates = {}
    if isinstance(out, dict):
        for key in ("key_img", "key_aud", "index_spikes"):
            value = out.get(key)
            if isinstance(value, torch.Tensor):
                output_rates[key + "_spike_rate"] = float((value.abs() > 1e-8).float().mean())
    record.update(status="ok", device=str(device), batch_size=batch_size,
                  parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  latency_ms_per_exposure=1000 * elapsed / max(1, repeats * batch_size),
                  dense_macs_per_batch=dense, active_macs_per_batch=active,
                  observed_modules=len(counter.rows), cuda_peak_memory_bytes=(
                      torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
                  **output_rates)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/v14pro")
    ap.add_argument("--repeats", type=int, default=8)
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error("--repeats must be positive")
    root = Path(args.root).resolve()
    configs = sorted(root.glob("seed_*/**/config.yaml"))
    records = [profile_one(path, args.repeats) for path in configs]
    output = root / "complexity_profile.json"
    output.write_text(json.dumps(dict(root=str(root), repeats=args.repeats, records=records),
                                 indent=2), encoding="utf-8")
    print(f"[profile] {len(records)} configurations -> {output}")
    for row in records:
        print(json.dumps({k: row[k] for k in ("experiment", "status", "parameters",
                                               "latency_ms_per_exposure") if k in row}))


if __name__ == "__main__":
    main()
