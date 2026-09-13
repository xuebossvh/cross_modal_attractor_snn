"""CPU smoke tests for the v12b multi-scale audio path."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import load_config
from models.network import CrossModalSNN
from scripts.train import _apply_trainable_prefixes, _masked_tf_grad_loss


def main():
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


if __name__ == "__main__":
    main()
