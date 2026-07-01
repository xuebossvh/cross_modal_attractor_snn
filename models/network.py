"""顶层跨模态 SNN 联想记忆网络（统一 cue->补全 接口）。"""

import torch
import torch.nn as nn

from .encoders import ImageSNNEncoder, AudioSNNEncoder
from .memory import CrossModalAttractorMemory
from .decoders import (ClassifierHead, ImageDecoder, AudioDecoder,
                       AudioResidualDecoder)
from .lif import rate


class CrossModalSNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["dims"]
        s = cfg["snn"]
        ab = cfg.get("ablation", {})
        self.T = s["T"]
        ac = cfg["audio"]
        self.audio_encoder_norm_mode = ac.get("encoder_norm_mode", "global")
        self.audio_encoder_local_mix = float(ac.get("encoder_local_mix", 0.5))

        self.img_encoder = ImageSNNEncoder(
            d["img_in"], d["img_hidden"], d["D_img"], self.T,
            s["beta"], s["v_threshold"], s["surrogate_alpha"],
            encoding=s.get("img_encoding", "first_spike_trace"),
            trace_decay=s.get("trace_decay", 0.9))

        aud_enc = s.get("aud_encoder", "conv")
        self.aud_encoder = AudioSNNEncoder(
            d["aud_in"], d["aud_hidden"], d["D_aud"], self.T,
            s["beta"], s["v_threshold"], s["surrogate_alpha"], s["encoding"],
            encoder_type=aud_enc,
            n_mels=ac["n_mels"], n_frames=ac["n_frames"],
            conv_ch1=s.get("aud_conv_ch1", 16),
            conv_ch2=s.get("aud_conv_ch2", 32))

        self.memory = CrossModalAttractorMemory(cfg)

        self.classifier = ClassifierHead(d["N_index"], d["num_classes"])
        self.image_decoder = ImageDecoder(d["N_value_img"])
        aud_dec_ch = s.get("aud_decoder_base_ch", 128)
        self.audio_decoder = AudioDecoder(
            d["N_value_aud"], ac["n_mels"], ac["n_frames"],
            base_ch=aud_dec_ch,
            start_hw=s.get("aud_decoder_start_hw", 4),
            refine_blocks=s.get("aud_decoder_refine_blocks", 0))

        detail_cfg = cfg.get("audio_detail", {})
        self.use_audio_detail_residual = detail_cfg.get("enabled", False)
        if self.use_audio_detail_residual:
            self.audio_residual_decoder = AudioResidualDecoder(
                d["D_aud"], ac["n_mels"], ac["n_frames"],
                hidden=detail_cfg.get("hidden", 256),
                scale=detail_cfg.get("scale", 0.35),
                zero_init=detail_cfg.get("zero_init", True))
        else:
            self.audio_residual_decoder = None

        self.use_audio_aux = ab.get("use_audio_aux_cls", True)
        if self.use_audio_aux:
            self.aux_aud_classifier = ClassifierHead(
                d["N_key_aud"], d["num_classes"])
        else:
            self.aux_aud_classifier = None

    def _normalize_audio_for_encoder(self, x_aud):
        if x_aud is None:
            return None
        mode = self.audio_encoder_norm_mode
        if mode in ("global", "dataset", "none", None):
            return x_aud

        flat = x_aud.flatten(1)
        lo = flat.min(dim=1).values.view(-1, 1, 1)
        hi = flat.max(dim=1).values.view(-1, 1, 1)
        per_sample = ((x_aud - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)

        if mode == "per_sample":
            return per_sample
        if mode == "hybrid":
            mix = max(0.0, min(1.0, self.audio_encoder_local_mix))
            return ((1.0 - mix) * x_aud + mix * per_sample).clamp(0.0, 1.0)
        raise ValueError(f"Unknown audio.encoder_norm_mode: {mode}")

    def forward(self, x_img_cue=None, x_aud_cue=None,
                x_img_target=None, x_aud_target=None,
                training_mode=False, phase="readout"):
        assert (x_img_cue is not None) or (x_aud_cue is not None), \
            "至少需要一种 cue 模态作为输入"

        if not training_mode:
            phase = "readout"
            x_img_target = None
            x_aud_target = None

        spike_img_cue = self.img_encoder(x_img_cue) if x_img_cue is not None else None
        spike_aud_cue = (self.aud_encoder(self._normalize_audio_for_encoder(x_aud_cue))
                         if x_aud_cue is not None else None)

        spike_img_tgt = None
        spike_aud_tgt = None
        if training_mode and phase == "binding":
            if x_img_target is not None:
                spike_img_tgt = self.img_encoder(x_img_target)
            if x_aud_target is not None:
                spike_aud_tgt = self.aud_encoder(
                    self._normalize_audio_for_encoder(x_aud_target))

        mem = self.memory(
            spike_img_cue=spike_img_cue, spike_aud_cue=spike_aud_cue,
            spike_img_target=spike_img_tgt, spike_aud_target=spike_aud_tgt,
            phase=phase)

        out = {
            "index_spikes": mem["index_spikes"],
            "index_state": mem["index_state"],
            "spike_img_cue": spike_img_cue, "spike_aud_cue": spike_aud_cue,
            "key_img": mem.get("key_img"), "key_aud": mem.get("key_aud"),
            "v_img_from_A": mem["v_img_from_A"], "v_aud_from_A": mem["v_aud_from_A"],
            "v_img_target": mem["v_img_target"], "v_aud_target": mem["v_aud_target"],
        }

        out["logits"] = self.classifier(mem["index_state"])
        out["aux_aud_logits"] = None
        if (self.aux_aud_classifier is not None
                and mem.get("key_aud") is not None):
            out["aux_aud_logits"] = self.aux_aud_classifier(rate(mem["key_aud"]))

        out["recovered_img"] = self.image_decoder(mem["v_img_from_A"])

        aud_base = self.audio_decoder(mem["v_aud_from_A"])
        aud_residual = None
        if self.audio_residual_decoder is not None and spike_aud_cue is not None:
            aud_detail = rate(spike_aud_cue)
            aud_residual = self.audio_residual_decoder(aud_detail)
            out["recovered_aud"] = (aud_base + aud_residual).clamp(0.0, 1.0)
        else:
            out["recovered_aud"] = aud_base
        out["recovered_aud_base"] = aud_base
        out["audio_residual"] = aud_residual
        return out

    @torch.no_grad()
    def infer(self, x_img_cue=None, x_aud_cue=None):
        self.eval()
        return self.forward(x_img_cue=x_img_cue, x_aud_cue=x_aud_cue,
                            training_mode=False, phase="readout")
