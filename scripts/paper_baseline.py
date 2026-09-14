"""Train clean-only external recognizers or mask-aware CNN recovery controls."""

import bootstrap  # noqa: F401
import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from common import (CUE_MODES, build_cue, load_config, resolve_train_corrupt_modes,
                    sample_cue_mode, select_targets, set_seed, unpack_paired_batch, save_checkpoint_atomic)
from data.dataset import build_loaders
from models.paper_baselines import CleanRecognizers, RecoveryCNN
from paths import resolve_from_root


def make_model(kind, cfg=None):
    if kind in ("recognizer", "classifier"):
        return CleanRecognizers()
    width = int((cfg or {}).get("paper", {}).get("ann_width", 32))
    conditioned = kind in ("conditioned_cnn", "matched_cnn")
    return RecoveryCNN(conditioned=conditioned, width=width)


def region_mse(rec, target, mask):
    if mask is None:
        return F.mse_loss(rec, target)
    return ((rec - target).square() * mask).flatten(1).sum(1).div(
        mask.flatten(1).sum(1).clamp_min(1)).mean()


def batch_loss(model, batch, cfg, kind, mode, family, device, protos):
    img, aud, labels, _ = unpack_paired_batch(batch)
    img, aud, labels = img.to(device), aud.to(device), labels.to(device)
    if kind in ("recognizer", "classifier"):
        li, la = model.img(img), model.aud(aud)
        loss = F.cross_entropy(li, labels) + F.cross_entropy(la, labels)
        accuracy = .5 * ((li.argmax(1) == labels).float().mean() +
                         (la.argmax(1) == labels).float().mean())
        return loss, float(1 - accuracy)
    ci, ca, masks = build_cue(img, aud, mode, cfg, severity=.4,
                             img_mode=family[0], aud_mode=family[1], return_masks=True)
    ti, ta, _, _ = select_targets(mode, img, aud, *protos, labels)
    logits, ri, ra = model(ci, ca, masks, *protos)
    loss = (F.cross_entropy(logits, labels) + F.mse_loss(ri, ti) +
            4 * F.mse_loss(ra, ta) + region_mse(ri, ti, masks.get("img")) +
            4 * region_mse(ra, ta, masks.get("aud")))
    return loss, float(loss.detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--kind", choices=("recognizer", "classifier", "cue_cnn", "conditioned_cnn", "matched_cnn"), required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg["train"].get("require_cuda", True) and not torch.cuda.is_available():
        raise RuntimeError("Publication training requires CUDA; use unit tests on CPU")
    device = torch.device(cfg["device"])
    set_seed(cfg["seed"] + (100000 if args.kind == "recognizer" else 0))
    train, val = build_loaders(cfg, eval_split="val")
    protos = (train.dataset.prototype_img.to(device), train.dataset.prototype_aud.to(device))
    model = make_model(args.kind, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    last, best = (resolve_from_root(cfg["train"][k]) for k in ("ckpt_path", "best_ckpt_path"))
    last.parent.mkdir(parents=True, exist_ok=True)
    start, best_score = 0, float("inf")
    if args.resume:
        state = torch.load(last, map_location=device)
        if state.get("kind") != args.kind:
            raise ValueError("Baseline resume kind mismatch")
        for section in ("paper", "train", "data", "audio"):
            if state.get("cfg", {}).get(section) != cfg.get(section):
                raise ValueError(f"Baseline resume configuration mismatch: {section}")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start, best_score = state["epoch"] + 1, state["best_score"]
    log_path = last.with_suffix(".training.jsonl")
    for epoch in range(start, cfg["train"]["epochs"]):
        # Epoch-local streams also make interrupted runs repeatable.
        set_seed(cfg["seed"] + epoch * 1009 + (100000 if args.kind == "recognizer" else 0))
        generator = train.generator or getattr(train.sampler, "generator", None)
        if generator is not None:
            generator.manual_seed(cfg["seed"] + epoch * 1009)
        import numpy as np
        train.dataset._rng = np.random.default_rng(cfg["seed"] + epoch * 1009)
        model.train()
        loss_sum, n, started = 0., 0, time.perf_counter()
        for step, batch in enumerate(train):
            family = resolve_train_corrupt_modes(cfg, epoch, step)
            loss, _ = batch_loss(model, batch, cfg, args.kind, sample_cue_mode(cfg), family, device, protos)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite baseline loss")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            count = len(batch[2])
            loss_sum += float(loss.detach()) * count
            n += count
        model.eval()
        score, count = 0., 0
        with torch.no_grad():
            for step, batch in enumerate(val):
                set_seed(20260915 + step)
                family = resolve_train_corrupt_modes(cfg, 0, step)
                _, value = batch_loss(model, batch, cfg, args.kind, "corrupt_both", family, device, protos)
                score += value * len(batch[2])
                count += len(batch[2])
        score /= count
        if args.kind not in ("recognizer", "classifier"):
            from scripts.paper_validation import validate_recovery
            score = validate_recovery(model, val, cfg, device, cnn=True)["score"]
        improved = score < best_score
        best_score = min(best_score, score)
        state = dict(model=model.state_dict(), opt=opt.state_dict(), epoch=epoch,
                     best_score=best_score, cfg=cfg, kind=args.kind)
        if improved:
            save_checkpoint_atomic(state, best)
        save_checkpoint_atomic(state, last)
        row = dict(epoch=epoch, train_loss=loss_sum/n, val_score=score,
                   seconds=time.perf_counter()-started, training_exposures=n)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
