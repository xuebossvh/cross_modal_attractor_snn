"""CPU smoke tests for the v12a audio reconstruction path."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import load_config
from models.network import CrossModalSNN
from scripts.train import _apply_trainable_prefixes


def main():
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
