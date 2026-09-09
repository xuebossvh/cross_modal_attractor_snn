"""Offline CPU regression checks. Optional --parent verifies real parent weights."""

import bootstrap  # noqa: F401

import argparse
import copy
import contextlib
import csv
import io
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch
import yaml

from common import load_config, select_targets
from models.network import CrossModalSNN
from models.frozen_base import (ADAPTER_PREFIXES, file_sha256, base_digest,
                                load_frozen_parent, frozen_metadata, verify_frozen_resume,
                                verify_audio_normalization)
from train import compute_losses, _build_train_optimizer, _checkpoint_payload
from evaluate import eval_mode, _paired_cross_metrics
from run_v11f_suite import build_jobs


def test_config_contract():
    cfg = load_config("configs/v11f.yaml")
    pairing = cfg["data"]["pairing"]

    assert cfg["data"]["dataset"] == "mnist_fsdd"
    assert cfg["data"]["use_mnist"] is True
    assert pairing["mode"] == "category_many_to_many"
    assert pairing["enabled"] is False
    assert pairing["return_pair_id"] is False
    assert pairing["sample_targets_for_missing"] is False
    assert cfg["cross_key_conditioning"]["enabled"] is True
    assert cfg["cross_key_conditioning"]["detach_key"] is True
    assert cfg["cross_detail_conditioning"]["enabled"] is False
    assert cfg["pair_alignment"]["enabled"] is False
    assert cfg["detail_conditioning"]["detach_value_for_recon"] is True
    assert cfg["train"]["eval_ckpt_path"] == cfg["train"]["ckpt_path"]
    assert cfg["validation"]["enabled"] is False
    assert cfg["data"]["batch_size"] == 128
    assert cfg["index"]["input_schedule"] == "simultaneous"
    assert cfg["train"]["freeze_base"] and cfg["train"]["require_cuda"]
    assert cfg["cross_key_conditioning"]["mode"] == "masked_feature"
    no_causal = load_config("configs/v11f_no_causal.yaml")
    assert not no_causal["cross_key_conditioning"]["causal_training"]["enabled"]
    assert no_causal["train"]["epochs"] == cfg["train"]["epochs"]
    return cfg


def test_category_targets():
    labels = torch.tensor([3, 7])
    clean_img = torch.rand(2, 1, 28, 28)
    clean_aud = torch.rand(2, 64, 64)
    proto_img = torch.rand(10, 1, 28, 28)
    proto_aud = torch.rand(10, 64, 64)

    img_only = select_targets(
        "clean_img_only", clean_img, clean_aud, proto_img, proto_aud,
        labels, paired_missing_targets=False)
    assert img_only[2:] == ("sample", "category")
    assert torch.equal(img_only[0], clean_img)
    assert torch.equal(img_only[1], proto_aud[labels])

    aud_only = select_targets(
        "clean_aud_only", clean_img, clean_aud, proto_img, proto_aud,
        labels, paired_missing_targets=False)
    assert aud_only[2:] == ("category", "sample")
    assert torch.equal(aud_only[0], proto_img[labels])
    assert torch.equal(aud_only[1], clean_aud)

    both = select_targets(
        "clean_both", clean_img, clean_aud, proto_img, proto_aud,
        labels, paired_missing_targets=False)
    assert both[2:] == ("sample", "sample")
    assert torch.equal(both[0], clean_img)
    assert torch.equal(both[1], clean_aud)


def check_invariants(model, parent):
    model.train()
    for name, module in model.named_modules():
        if not name.startswith(ADAPTER_PREFIXES) and name not in (
                "img_cross_adapter", "aud_cross_adapter"):
            assert not module.training, name
    assert all(p.requires_grad == name.startswith(ADAPTER_PREFIXES)
               for name, p in model.named_parameters())
    before = base_digest(model)
    image, audio = torch.rand(2, 1, 28, 28), torch.rand(2, 64, 64)
    im, am = torch.zeros_like(image), torch.zeros_like(audio)
    im[..., 7:18, 8:20], am[..., 20:42] = 1, 1
    cues = dict(x_img_cue=image * (1-im), x_aud_cue=audio * (1-am),
                img_cue_mask=im, aud_cue_mask=am)
    keys = dict(cross_key_img_rate_override=torch.rand(2, 128),
                cross_key_aud_rate_override=torch.rand(2, 128))
    with torch.no_grad():
        reference = parent(**cues)
        initial = model(**cues, **keys)
    for name in ("index_state", "logits", "recovered_img", "recovered_aud"):
        assert torch.equal(initial[name], reference[name]), ("initial", name)
    opt = _build_train_optimizer(model, model.cfg)
    for step in range(3):
        opt.zero_grad()
        out = model(**cues, **keys)
        loss = ((torch.sigmoid(out["recovered_img"]) - image).square().mean()
                + (out["recovered_aud"] - audio).square().mean())
        loss.backward()
        for adapter in (model.img_cross_adapter, model.aud_cross_adapter):
            assert adapter.out.weight.grad.abs().sum() > 0
            if step > 0:
                assert adapter.gate.weight.grad.abs().sum() > 0
                assert adapter.key.weight.grad.abs().sum() > 0
        assert all(p.grad is None for n, p in model.named_parameters()
                   if not n.startswith(ADAPTER_PREFIXES))
        opt.step()
    with torch.no_grad():
        normal = model(**cues, **keys)
        zero = model(**cues, disable_img_to_aud_cross=True, disable_aud_to_img_cross=True)
        wrong = model(**cues, **{k: v.flip(0) for k, v in keys.items()})
        zero_key = model(**cues, **{k: torch.zeros_like(v) for k, v in keys.items()})
        for name in ("recovered_img", "recovered_aud", "index_state", "logits"):
            assert torch.equal(zero[name], reference[name]), ("zero", name)
            assert torch.equal(zero_key[name], reference[name]), ("zero_key", name)
        for candidate in (normal, wrong):
            assert torch.equal(candidate["logits"], reference["logits"])
            assert torch.equal(candidate["recovered_img"][im == 0], reference["recovered_img"][im == 0])
            assert torch.equal(candidate["recovered_aud"][am == 0], reference["recovered_aud"][am == 0])
        assert not torch.equal(normal["recovered_img"][im.bool()], wrong["recovered_img"][im.bool()])
        assert not torch.equal(normal["recovered_aud"][am.bool()], wrong["recovered_aud"][am.bool()])
        for cue in ({"x_img_cue": image}, {"x_aud_cue": audio},
                    {"x_img_cue": image, "x_aud_cue": audio}):
            output, baseline = model(**cue, **keys), parent(**cue)
            assert torch.equal(output["logits"], baseline["logits"])
            if "x_img_cue" in cue:
                assert torch.equal(output["recovered_img"], baseline["recovered_img"])
            if "x_aud_cue" in cue:
                assert torch.equal(output["recovered_aud"], baseline["recovered_aud"])
    assert before == base_digest(model)
    frozen_metadata(model)
    state = _checkpoint_payload(model, opt, None, model.cfg, 0)
    verify_frozen_resume(model, state)
    state["frozen_base"]["base_sha256"] = "incorrect"
    try:
        verify_frozen_resume(model, state)
        raise AssertionError("tampered base digest accepted")
    except RuntimeError:
        pass
    model.frozen_base_digest = before
    print("PASS parent equality, spatial masks, both-direction gradients, frozen state, resume")


def check_losses_and_eval(model):
    image, audio = torch.rand(4, 1, 28, 28), torch.rand(4, 64, 64)
    labels = torch.tensor([0, 0, 1, 1])
    pi, pa = torch.rand(10, 1, 28, 28), torch.rand(10, 64, 64)
    opt = _build_train_optimizer(model, model.cfg)
    modes = ["corrupt_img_only", "corrupt_aud_only", "corrupt_both",
             "clean_img_corrupt_aud", "corrupt_img_clean_aud",
             "clean_img_only", "clean_aud_only", "clean_both"]
    for i, mode in enumerate(modes):
        opt.zero_grad()
        loss, logs = compute_losses(model, image, audio, labels, mode,
                                    model.cfg, pi, pa, step=i)
        assert torch.isfinite(loss), mode
        if mode == "clean_both":
            assert not loss.requires_grad
        else:
            assert loss.requires_grad and "cross_pair" in logs, mode
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters()
                       if p.grad is not None)
            opt.step()
    loss, logs = compute_losses(model, image, audio, labels * 0,
                                "corrupt_both", model.cfg, pi, pa)
    assert torch.isfinite(loss) and logs["img2aud_wrong_n"] == 0
    assert logs["img2aud_n"] == 4 and loss.requires_grad
    frozen_metadata(model)
    for protocol in ("fixed_mask", "legacy_random"):
        result = eval_mode(model, [(image, audio, labels)], model.cfg,
                           "corrupt_both", torch.device("cpu"), 0.4, pi, pa,
                           max_batches=1, protocol=protocol, cross_key_mode="sweep")
        for variant in ("normal", "zero", "wrong", "same_class"):
            assert result["cross_counts"][f"content_aud_{variant}_acc"] == 4
            assert f"img2aud_{variant}_mse" in result["cross_attr"]
        assert 0 <= result["cross_attr"]["img2aud_win_both"] <= 1
    print("PASS 8 cue losses, same-class-only batch, fixed/random sweep and content metrics")


def check_cli(cfg, model, parent_path, temp):
    import train
    import evaluate
    import demo_inference
    from PIL import Image, ImageStat
    from torch.utils.data import DataLoader, TensorDataset

    cfg = copy.deepcopy(cfg)
    cfg["train"].update(require_cuda=False, init_ckpt_path=str(parent_path),
                        ckpt_path=str(temp / "trained.pt"),
                        eval_ckpt_path=str(temp / "trained.pt"))
    dataset = TensorDataset(torch.rand(4, 1, 28, 28), torch.rand(4, 64, 64),
                            torch.tensor([0, 0, 1, 1]))
    dataset.prototype_img = torch.rand(10, 1, 28, 28)
    dataset.prototype_aud = torch.rand(10, 64, 64)
    dataset.use_real_audio = True
    loader = DataLoader(dataset, batch_size=4)

    def loaders(config, **kwargs):
        config["_audio_norm_stats"] = model.parent_audio_norm_stats
        return loader, loader

    config_path = temp / "cli.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    tables, figures = temp / "tables", temp / "figures"
    tables.mkdir()
    figures.mkdir()
    capture = io.StringIO()
    with contextlib.ExitStack() as stack:
        stack.enter_context(contextlib.redirect_stdout(capture))
        stack.enter_context(contextlib.redirect_stderr(capture))
        for module in (train, evaluate, demo_inference):
            stack.enter_context(patch.object(module, "build_loaders", loaders))
        for module in (train, demo_inference):
            stack.enter_context(patch.object(module, "ensure_output_dirs", lambda cfg: None))
        stack.enter_context(patch.object(evaluate, "tables_dir", lambda cfg: tables))
        stack.enter_context(patch.object(demo_inference, "tables_dir", lambda cfg: tables))
        stack.enter_context(patch.object(demo_inference, "figures_dir", lambda cfg: figures))
        stack.enter_context(patch.object(evaluate, "EVAL_MODES", ["corrupt_both"]))
        for argv in (["train.py", "--config", str(config_path), "--epochs", "1"],
                     ["train.py", "--config", str(config_path), "--epochs", "2", "--resume"]):
            with patch.object(sys, "argv", argv):
                train.main()
        for protocol in ("fixed_mask", "legacy_random"):
            with patch.object(sys, "argv", ["evaluate.py", "--config", str(config_path),
                                            "--cross_key", "sweep", "--protocol", protocol,
                                            "--max_batches", "1"]):
                evaluate.main()
            with patch.object(sys, "argv", ["demo_inference.py", "--config", str(config_path),
                                            "--protocol", protocol, "--num", "2"]):
                demo_inference.main()
        with patch.object(sys, "argv", ["demo_inference.py", "--config", str(config_path),
                                        "--ckpt", str(temp / "missing.pt")]):
            try:
                demo_inference.main()
                raise AssertionError("missing checkpoint produced random demo")
            except SystemExit:
                pass
    state = torch.load(temp / "trained.pt", map_location="cpu")
    assert state["epoch"] == 1 and state["frozen_base"]["base_sha256"] == base_digest(model)
    try:
        verify_audio_normalization(model, {"_audio_norm_stats": {"lo": 999}})
        raise AssertionError("changed normalization accepted")
    except RuntimeError:
        pass
    csv_files = sorted(tables.glob("eval_*.csv"))
    assert len(csv_files) == 2
    for path in csv_files:
        with path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert any(row["metric"] == "content_aud_normal_acc" for row in rows)
        assert any(row["metric"] == "img2aud_win_both" for row in rows)
        assert all(int(row["n"]) == 4 for row in rows if row["metric"].startswith("content_"))
    images = sorted(figures.glob("demo_*.png"))
    assert len(images) == 6
    for path in images:
        with Image.open(path) as image:
            assert min(image.size) > 100 and max(ImageStat.Stat(image.convert("RGB")).stddev) > 1
    print("PASS CLI train/resume, 2 protocol CSVs, 6 nonblank demos, missing checkpoint rejection")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=Path, default=None)
    ap.add_argument("--cli", action="store_true", help="Also test entry points on synthetic data")
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    cfg = test_config_contract()
    test_category_targets()
    if args.parent is None:
        cfg["snn"]["T"] = 2
    cfg["device"] = "cpu"
    parent_cfg = copy.deepcopy(cfg)
    parent_cfg["train"]["freeze_base"] = False
    parent_cfg["cross_key_conditioning"].update(enabled=False, mode="value_residual")
    parent = CrossModalSNN(parent_cfg).eval()
    with tempfile.TemporaryDirectory(prefix="v11f-test-") as temp:
        path = args.parent or Path(temp) / "parent.pt"
        if args.parent is None:
            torch.save({"model": parent.state_dict(), "cfg": parent_cfg}, path)
            cfg["train"]["parent_sha256"] = file_sha256(path)
        state = torch.load(path, map_location="cpu")
        parent.load_state_dict(state["model"])
        model = CrossModalSNN(cfg)
        load_frozen_parent(model, state, path)
        del state
        check_invariants(model, parent)
        check_losses_and_eval(model)
        if args.cli:
            check_cli(cfg, model, path.resolve(), Path(temp))
    jobs = build_jobs(["configs/v11f.yaml", "configs/v11f_control.yaml",
                       "configs/v11f_no_causal.yaml"])
    assert sum(c[0] == "scripts/train.py" for c, _ in jobs) == 2
    assert sum(c[0] == "scripts/demo_inference.py" for c, _ in jobs) == 6
    assert len({str(log) for _, log in jobs}) == len(jobs)
    print("PASS suite order/configs; v11f regression complete (CPU, not GPU convergence)")


if __name__ == "__main__":
    main()
