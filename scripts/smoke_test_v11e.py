"""CPU-only v11e checks for real-pair manifests and instance alignment."""

import bootstrap  # noqa: F401

import csv
import tempfile
from pathlib import Path

import torch

from common import load_config, select_targets
from data.dataset import TruePairedManifestDataset, build_loaders
from models.network import CrossModalSNN
from evaluate import _global_pair_retrieval
from train import _pair_alignment_loss, _validate_model


FIELDS = [
    "pair_id", "source_id", "image_source_id", "audio_source_id",
    "speaker_id", "split", "label", "image_path", "audio_path",
]


def _make_manifest(root):
    rows = []
    pair_id = 0
    for split, speakers in {
            "train": ("s1", "s2"),
            "val": ("s3", "s4"),
            "test": ("s5", "s6")}.items():
        for speaker in speakers:
            for label in range(10):
                source = f"{speaker}/u{label}"
                image_rel = Path("images") / speaker / f"u{label}.pt"
                audio_rel = Path("audio") / speaker / f"u{label}.pt"
                (root / image_rel).parent.mkdir(parents=True, exist_ok=True)
                (root / audio_rel).parent.mkdir(parents=True, exist_ok=True)
                torch.save(torch.rand(1, 28, 28), root / image_rel)
                torch.save(torch.rand(64, 64), root / audio_rel)
                rows.append({
                    "pair_id": pair_id,
                    "source_id": source,
                    "image_source_id": source,
                    "audio_source_id": source,
                    "speaker_id": speaker,
                    "split": split,
                    "label": label,
                    "image_path": image_rel.as_posix(),
                    "audio_path": audio_rel.as_posix(),
                })
                pair_id += 1
    path = root / "pairs.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _small_cfg(path, manifest):
    cfg = load_config(path)
    cfg["device"] = "cpu"
    cfg["snn"]["T"] = 2
    cfg["data"]["manifest_path"] = str(manifest)
    cfg["data"]["num_workers"] = 0
    cfg["data"]["batch_size"] = 4
    cfg["data"]["train_samples_per_epoch"] = 20
    cfg["data"]["prototype_candidates_per_class"] = 1
    return cfg


def test_manifest_and_loader(root):
    manifest = _make_manifest(root)
    cfg = _small_cfg("configs/v11e.yaml", manifest)
    train_loader, val_loader = build_loaders(cfg, eval_split="val")
    assert len(train_loader.dataset) == 20
    assert len(val_loader.dataset) == 20
    assert len(train_loader) == 5
    batch = next(iter(val_loader))
    assert len(batch) == 4
    assert batch[0].shape == (4, 1, 28, 28)
    assert batch[1].shape == (4, 64, 64)

    targets = select_targets(
        "clean_aud_only", batch[0], batch[1],
        train_loader.dataset.prototype_img,
        train_loader.dataset.prototype_aud, batch[2],
        paired_missing_targets=True, paired_target_kind="paired-sample")
    assert targets[2:] == ("paired-sample", "paired-sample")
    assert torch.equal(targets[0], batch[0])
    return cfg, val_loader


def test_pair_alignment(cfg, val_loader):
    model = CrossModalSNN(cfg)
    batch = 4
    image = torch.rand(batch, 1, 28, 28)
    audio = torch.rand(batch, 64, 64)
    labels = torch.tensor([1, 1, 2, 2])
    pair_ids = torch.tensor([10, 11, 12, 13])
    out = model(x_img_cue=image, x_aud_cue=audio)
    assert out["img_pair_embedding"].shape == (batch, 128)
    assert out["aud_pair_embedding"].shape == (batch, 128)
    loss, logs = _pair_alignment_loss(out, labels, pair_ids, cfg)
    assert torch.isfinite(loss) and loss.item() > 0
    assert logs["pair_align_n"] == 4.0
    perfect_i2a, perfect_a2i = _global_pair_retrieval(
        out["img_pair_embedding"].detach(),
        out["img_pair_embedding"].detach(), labels, pair_ids)
    assert perfect_i2a == 1.0 and perfect_a2i == 1.0

    cfg["validation"]["max_batches"] = 1
    metrics = _validate_model(model, val_loader, cfg, torch.device("cpu"))
    assert metrics is not None and torch.isfinite(
        torch.tensor(metrics["score"]))

    control_cfg = load_config("configs/v11e_control.yaml")
    control_cfg["device"] = "cpu"
    control_cfg["snn"]["T"] = 2
    control = CrossModalSNN(control_cfg)
    control.load_state_dict(model.state_dict(), strict=True)
    assert not control.use_cross_detail_conditioning
    assert not control.use_pair_alignment


def test_rejects_fake_pair(cfg, root):
    source = Path(cfg["data"]["manifest_path"])
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    rows[0]["audio_source_id"] = "different/utterance"
    invalid = root / "invalid.csv"
    with invalid.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    bad_cfg = dict(cfg)
    bad_cfg["data"] = dict(cfg["data"])
    bad_cfg["data"]["manifest_path"] = str(invalid)
    try:
        TruePairedManifestDataset(bad_cfg, split="train")
    except ValueError as error:
        assert "not from the same source" in str(error)
    else:
        raise AssertionError("mismatched image/audio source ids were accepted")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        config, validation_loader = test_manifest_and_loader(root)
        test_pair_alignment(config, validation_loader)
        test_rejects_fake_pair(config, root)
    print("v11e smoke test passed")
