"""CPU-only checks for the v11e category-level MNIST/FSDD protocol."""

import bootstrap  # noqa: F401

import torch

from common import load_config, select_targets
from models.network import CrossModalSNN


def test_config_contract():
    cfg = load_config("configs/v11e.yaml")
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


def test_main_control_structure(cfg):
    cfg = dict(cfg)
    cfg["snn"] = dict(cfg["snn"])
    cfg["snn"]["T"] = 2
    main = CrossModalSNN(cfg)

    control_cfg = load_config("configs/v11e_control.yaml")
    control_cfg["device"] = "cpu"
    control_cfg["snn"]["T"] = 2
    control = CrossModalSNN(control_cfg)

    assert main.use_cross_key_conditioning
    assert not control.use_cross_key_conditioning
    assert not main.use_cross_detail_conditioning
    assert not main.use_pair_alignment
    control.load_state_dict(main.state_dict(), strict=True)

    image = torch.rand(2, 1, 28, 28)
    audio = torch.rand(2, 64, 64)
    with torch.no_grad():
        out = main(x_img_cue=image, x_aud_cue=audio)
    assert out["logits"].shape == (2, 10)
    assert out["recovered_img"].shape == (2, 1, 28, 28)
    assert out["recovered_aud"].shape == (2, 64, 64)
    assert out.get("img_pair_embedding") is None
    assert out.get("aud_pair_embedding") is None


if __name__ == "__main__":
    config = test_config_contract()
    test_category_targets()
    test_main_control_structure(config)
    print("v11e category-binding smoke test passed")
