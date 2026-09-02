"""CPU-only v11d smoke test: fixed pairs, parent compatibility, Cross-Detail."""

import bootstrap  # noqa: F401

import torch

from common import load_config, select_targets
from data.dataset import PairedAudioVisualDataset
from models.network import CrossModalSNN


def _small_cfg(path):
    cfg = load_config(path)
    cfg["device"] = "cpu"
    cfg["snn"]["T"] = 2
    return cfg


def test_fixed_pairs():
    cfg = _small_cfg("configs/v11d.yaml")
    cfg["audio"]["use_real_audio"] = False
    cfg["data"]["use_mnist"] = False
    cfg["data"]["train_subset"] = 64
    cfg["data"]["pairing"]["manifest_dir"] = ""
    ds = PairedAudioVisualDataset(cfg, train=True)
    first = ds[0]
    repeated = ds[0]
    assert len(first) == 4
    assert first[3] == repeated[3]
    assert torch.equal(first[1], repeated[1])
    pair_ids = [int(ds[i][3]) for i in range(len(ds))]
    assert len(set(pair_ids)) == len(pair_ids)

    by_label = {}
    found_distinct = False
    for i in range(len(ds)):
        sample = ds[i]
        label = int(sample[2])
        if label in by_label:
            other = ds[by_label[label]]
            assert sample[3] != other[3]
            assert not torch.equal(sample[1], other[1])
            found_distinct = True
            break
        by_label[label] = i
    assert found_distinct


def test_parent_and_cross_detail():
    parent = CrossModalSNN(_small_cfg("configs/v11c.yaml"))
    child = CrossModalSNN(_small_cfg("configs/v11d.yaml"))
    incompatible = child.load_state_dict(parent.state_dict(), strict=False)
    expected_prefixes = (
        "aud_to_img_detail_proj.", "img_to_aud_detail_proj.",
        "aud_to_img_detail_gate.", "img_to_aud_detail_gate.",
    )
    assert incompatible.missing_keys
    assert all(k.startswith(expected_prefixes) for k in incompatible.missing_keys)
    assert not incompatible.unexpected_keys

    batch = 3
    with torch.no_grad():
        child.aud_to_img_detail_proj.weight.normal_(0.0, 0.01)
        child.img_to_aud_detail_proj.weight.normal_(0.0, 0.01)
    base_img = torch.rand(batch, child.cfg["dims"]["N_value_img"])
    own_img = torch.rand(batch, child.cfg["dims"]["D_img"])
    aud_detail = torch.rand(batch, child.cfg["dims"]["aud_hidden"])
    fused, stats = child._fuse_decoder_state(
        base_img, own_img, "img", raw_cross_detail=aud_detail,
        return_cross_stats=True)
    disabled, _ = child._fuse_decoder_state(
        base_img, own_img, "img", raw_cross_detail=aud_detail,
        disable_cross_detail=True, return_cross_stats=True)
    assert fused.shape == disabled.shape
    assert stats["detail_gate"].shape == (batch, child.img_detail_dim)
    assert not torch.equal(fused, disabled)

    x_img = torch.rand(batch, 1, 28, 28)
    x_aud = torch.rand(batch, 64, 64)
    out = child(x_img_cue=x_img, x_aud_cue=x_aud)
    assert out["recovered_img"].shape == x_img.shape
    assert out["recovered_aud"].shape == x_aud.shape
    assert "recovered_aud_coarse" not in out


def test_paired_targets():
    batch = 2
    img = torch.rand(batch, 1, 28, 28)
    aud = torch.rand(batch, 64, 64)
    labels = torch.tensor([1, 2])
    proto_img = torch.rand(10, 1, 28, 28)
    proto_aud = torch.rand(10, 64, 64)
    tgt_img, tgt_aud, img_kind, aud_kind = select_targets(
        "clean_aud_only", img, aud, proto_img, proto_aud, labels,
        paired_missing_targets=True)
    assert torch.equal(tgt_img, img)
    assert torch.equal(tgt_aud, aud)
    assert (img_kind, aud_kind) == ("sample", "sample")


if __name__ == "__main__":
    test_fixed_pairs()
    test_parent_and_cross_detail()
    test_paired_targets()
    print("v11d smoke test passed")
