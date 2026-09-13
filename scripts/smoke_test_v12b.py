"""CPU smoke tests for the v12b multi-scale audio path."""

import sys
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import CUE_MODES, build_cue, load_config
from models.network import CrossModalSNN
from scripts.run_v12b_suite import build_jobs, select_configs
from scripts.train import _apply_trainable_prefixes, _masked_tf_grad_loss


def check_all_cues(model, cfg, name, compare_legacy=False):
    image = torch.rand(1, 1, 28, 28)
    audio = torch.rand(1, 64, 64)
    encoder_forward = model.aud_encoder.forward_with_local

    def legacy_forward(cue):
        key, detail, local = encoder_forward(cue)
        return key, detail, local["conv2"]

    for mode in CUE_MODES:
        img, aud, masks = build_cue(
            image, audio, mode, cfg, severity=0.4, return_masks=True)
        inputs = dict(x_img_cue=img, x_aud_cue=aud,
                      img_cue_mask=masks["img"], aud_cue_mask=masks["aud"])
        actual = model.infer(**inputs)
        assert actual["recovered_img"].shape == image.shape, (name, mode)
        assert actual["recovered_aud"].shape == audio.shape, (name, mode)
        for field in ("recovered_img", "recovered_aud", "index_state"):
            assert torch.isfinite(actual[field]).all(), (name, mode, field)
        if aud is not None and masks["aud"] is not None:
            visible = masks["aud"] == 0
            assert torch.equal(actual["recovered_aud"][visible], aud[visible])
        if compare_legacy:
            with patch.object(model.aud_encoder, "forward_with_local",
                              side_effect=legacy_forward):
                expected = model.infer(**inputs)
            for field in ("recovered_img", "recovered_aud", "index_state"):
                assert torch.equal(actual[field], expected[field]), (mode, field)
    print(f"PASS {name}: all {len(CUE_MODES)} cue modes"
          + ("; control matches legacy tensor path" if compare_legacy else ""))


def check_suite_selection():
    assert [p.stem for p in select_configs()] == ["v12b", "v12b_control"]
    selected = select_configs(True, "control")
    assert [p.stem for p in selected] == ["v12b_control", "v12b_no_causal"]
    jobs = build_jobs(selected)
    training = [command for command, _ in jobs if command[0] == "scripts/train.py"]
    assert len(training) == 1 and Path(training[0][2]).stem == "v12b_no_causal"
    assert Path(jobs[0][0][2]).stem == "v12b_control"
    assert not any(command[0] == "scripts/train.py"
                   for command, _ in build_jobs(selected, eval_only=True))
    assert [p.stem for p in select_configs(True, "no_causal")] == ["v12b_no_causal"]
    try:
        select_configs(False, "no_causal")
    except ValueError:
        pass
    else:
        raise AssertionError("Missing ablation flag must be rejected")
    print("PASS suite starts at control; only no-causal trains")


def main():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    cfg = load_config(ROOT / "configs/v12b.yaml")
    cfg["device"] = "cpu"
    model = CrossModalSNN(cfg).to("cpu")
    model.eval()

    assert model.audio_local_cue_multiscale is not None
    assert set(model.audio_local_cue_multiscale) == {"16", "32", "64"}
    assert model.aud_cross_adapter_mid is not None

    batch = 2
    image = torch.rand(batch, 1, 28, 28)
    audio = torch.rand(batch, 64, 64)
    image_mask = torch.zeros_like(image)
    audio_mask = torch.zeros_like(audio)
    audio_mask[:, :, 16:32] = 1.0
    output = model.infer(
        x_img_cue=image,
        x_aud_cue=audio,
        img_cue_mask=image_mask,
        aud_cue_mask=audio_mask,
    )
    assert output["recovered_img"].shape == image.shape
    assert output["recovered_aud"].shape == audio.shape
    assert torch.isfinite(output["recovered_aud"]).all()
    assert torch.allclose(
        output["recovered_aud"][audio_mask == 0], audio[audio_mask == 0],
        atol=1e-6)

    _, _, local = model.aud_encoder.forward_with_local(audio)
    assert local["conv1"].shape[:4] == (20, batch, 32, 32)
    assert local["conv2"].shape[:4] == (20, batch, 64, 16)

    trainable = _apply_trainable_prefixes(model, cfg)
    assert trainable
    allowed = (
        "audio_decoder.", "audio_local_cue_projector.",
        "audio_local_cue_multiscale.", "img_cross_adapter.",
        "aud_cross_adapter.", "aud_cross_adapter_mid.")
    assert all(name.startswith(allowed) for name, param in model.named_parameters()
               if param.requires_grad)

    rec = torch.rand(2, 4, 5)
    target = torch.rand_like(rec)
    mask = torch.zeros_like(rec)
    mask[:, :, 2] = 1.0
    assert torch.isfinite(_masked_tf_grad_loss(rec, target, mask))
    print("PASS v12b CPU multi-scale cue/mid-Cross-Key/pasteback smoke")
    check_all_cues(model, cfg, "v12b")
    for name in ("v12b_control", "v12b_no_causal"):
        test_cfg = load_config(ROOT / f"configs/{name}.yaml")
        test_cfg["device"] = "cpu"
        test_model = CrossModalSNN(test_cfg).eval()
        if name == "v12b_control":
            # Exercise the trained-projector case instead of an all-zero branch.
            with torch.no_grad():
                test_model.audio_local_cue_projector.body[-1].weight.normal_(0, 0.01)
        check_all_cues(test_model, test_cfg, name,
                       compare_legacy=(name == "v12b_control"))
    check_suite_selection()


if __name__ == "__main__":
    main()
