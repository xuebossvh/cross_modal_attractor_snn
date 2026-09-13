"""CPU smoke tests for the v12a audio reconstruction path."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import load_config
from models.network import CrossModalSNN
from scripts.demo_inference import _sample_demo_batch
from scripts.train import _apply_trainable_prefixes


def check_control_checkpoint_loading():
    parent_cfg = load_config(ROOT / "configs/v12a.yaml")
    # v11g contains Cross-Key adapters, but no v12a local-cue projector.
    parent_cfg["audio_local_cue"]["enabled"] = False
    parent = CrossModalSNN(parent_cfg)
    control_cfg = load_config(ROOT / "configs/v12a_control.yaml")
    control = CrossModalSNN(control_cfg)
    control.load_state_dict(parent.state_dict(), strict=True)
    assert control.img_cross_adapter is not None
    assert control.aud_cross_adapter is not None
    assert not control.use_cross_key_conditioning
    assert not control.use_audio_local_cue
    print("PASS v12a control strict loading with v11g-shaped adapter weights")


def check_demo_sampling():
    dataset = torch.utils.data.TensorDataset(
        torch.arange(100, dtype=torch.float32).view(100, 1, 1, 1),
        torch.arange(100, dtype=torch.float32).view(100, 1, 1),
        torch.arange(100, dtype=torch.long),
    )
    _, _, labels_a, _, indices_a = _sample_demo_batch(dataset, 10, 4321)
    _, _, labels_b, _, indices_b = _sample_demo_batch(dataset, 10, 4321)
    assert indices_a == indices_b
    assert len(indices_a) == 10 and len(set(indices_a)) == 10
    assert all(0 <= index < len(dataset) for index in indices_a)
    assert indices_a != list(range(10))
    assert torch.equal(labels_a, labels_b)
    print("PASS v12a demo fixed-seed full-test-set sampling")


def main():
    check_control_checkpoint_loading()
    check_demo_sampling()
    cfg = load_config(ROOT / "configs/v12a.yaml")
    cfg["device"] = "cpu"
    model = CrossModalSNN(cfg).to("cpu")
    model.eval()

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
    visible = audio_mask == 0
    assert torch.allclose(
        output["recovered_aud"][visible], audio[visible], atol=1e-6)

    local = torch.rand(20, batch, 64, 16, 16)
    key, instance, local_rate = model.aud_encoder.forward_with_local(audio)
    assert key.shape[:2] == (20, batch)
    assert instance.shape[:2] == (20, batch)
    assert local_rate.shape == (20, batch, 64, 16, 16)

    trainable = _apply_trainable_prefixes(model, cfg)
    assert trainable
    assert all(
        name.startswith(("audio_decoder.", "audio_local_cue_projector.",
                         "img_cross_adapter.", "aud_cross_adapter."))
        for name, param in model.named_parameters() if param.requires_grad
    )
    print("PASS v12a CPU shape/pasteback/local-cue/trainable-prefix smoke")


if __name__ == "__main__":
    main()
