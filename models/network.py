"""顶层跨模态 SNN 联想记忆网络（统一 cue->补全 接口）。"""

import torch
import torch.nn as nn

from .encoders import ImageSNNEncoder, AudioSNNEncoder
from .memory import CrossModalAttractorMemory
from .decoders import ClassifierHead, ImageDecoder, AudioDecoder, AudioRefiner
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
        detail_cfg = cfg.get("detail_conditioning", {})
        self.use_detail_conditioning = detail_cfg.get("enabled", False)
        self.detail_conditioning_detach = detail_cfg.get("detach", True)
        self.detail_conditioning_zero_missing = detail_cfg.get("zero_missing", True)
        self.detach_value_for_recon = detail_cfg.get("detach_value_for_recon", False)
        self.detail_fusion = detail_cfg.get("fusion", "concat")
        self.img_detail_dim = int(detail_cfg.get("img_detail_dim", d["D_img"]))
        self.aud_detail_dim = int(detail_cfg.get("aud_detail_dim", d["D_aud"]))

        img_decoder_in = d["N_value_img"]
        aud_decoder_in = d["N_value_aud"]
        if self.use_detail_conditioning:
            img_decoder_in += self.img_detail_dim
            aud_decoder_in += self.aud_detail_dim
            self.img_detail_projector = nn.Sequential(
                nn.Linear(d["D_img"], self.img_detail_dim),
                nn.LayerNorm(self.img_detail_dim),
                nn.ReLU(inplace=True),
            )
            self.aud_detail_projector = nn.Sequential(
                nn.Linear(d["D_aud"], self.aud_detail_dim),
                nn.LayerNorm(self.aud_detail_dim),
                nn.ReLU(inplace=True),
            )
            if self.detail_fusion == "gated_concat":
                self.img_detail_gate = nn.Sequential(
                    nn.Linear(d["N_value_img"] + self.img_detail_dim,
                              self.img_detail_dim),
                    nn.Sigmoid(),
                )
                self.aud_detail_gate = nn.Sequential(
                    nn.Linear(d["N_value_aud"] + self.aud_detail_dim,
                              self.aud_detail_dim),
                    nn.Sigmoid(),
                )
            elif self.detail_fusion != "concat":
                raise ValueError(f"Unknown detail_conditioning.fusion: {self.detail_fusion}")

        self.image_decoder = ImageDecoder(img_decoder_in)
        aud_dec_ch = s.get("aud_decoder_base_ch", 128)
        self.audio_decoder = AudioDecoder(
            aud_decoder_in, ac["n_mels"], ac["n_frames"],
            base_ch=aud_dec_ch,
            start_hw=s.get("aud_decoder_start_hw", 4),
            refine_blocks=s.get("aud_decoder_refine_blocks", 0),
            refine_type=s.get("aud_refine_type", "plain"))

        refiner_cfg = cfg.get("audio_refiner", {})
        self.use_audio_refiner = refiner_cfg.get("enabled", False)
        if self.use_audio_refiner:
            self.audio_refiner = AudioRefiner(
                ac["n_mels"], ac["n_frames"],
                hidden_ch=int(refiner_cfg.get("hidden_ch", 32)),
                blocks=int(refiner_cfg.get("blocks", 2)),
                delta_scale=float(refiner_cfg.get("delta_scale", 1.0)))
        else:
            self.audio_refiner = None

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

    def _cue_detail_state(self, spikes, dim, batch, device, dtype):
        if not self.use_detail_conditioning:
            return None
        if spikes is None:
            if self.detail_conditioning_zero_missing:
                return torch.zeros(batch, dim, device=device, dtype=dtype)
            raise ValueError(
                "detail_conditioning.zero_missing must be true when a cue is absent.")
        detail = rate(spikes)
        if self.detail_conditioning_detach:
            detail = detail.detach()
        return detail

    def _fuse_decoder_state(self, value_state, raw_detail, modality):
        if self.detach_value_for_recon:
            value_state = value_state.detach()
        if not self.use_detail_conditioning:
            return value_state

        if modality == "img":
            detail = self.img_detail_projector(raw_detail)
            if self.detail_fusion == "gated_concat":
                gate = self.img_detail_gate(torch.cat([value_state, detail], dim=1))
                detail = gate * detail
        elif modality == "aud":
            detail = self.aud_detail_projector(raw_detail)
            if self.detail_fusion == "gated_concat":
                gate = self.aud_detail_gate(torch.cat([value_state, detail], dim=1))
                detail = gate * detail
        else:
            raise ValueError(f"Unknown modality: {modality}")
        return torch.cat([value_state, detail], dim=1)

    def forward(self, x_img_cue=None, x_aud_cue=None,
                x_img_target=None, x_aud_target=None,
                training_mode=False, phase="readout", aud_cue_mask=None):
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

        img_dec_state = mem["v_img_from_A"]
        aud_dec_state = mem["v_aud_from_A"]
        img_detail = None
        aud_detail = None
        if self.use_detail_conditioning:
            batch = mem["index_state"].size(0)
            device = mem["index_state"].device
            dtype = mem["index_state"].dtype
            img_detail = self._cue_detail_state(
                spike_img_cue, self.cfg["dims"]["D_img"], batch, device, dtype)
            aud_detail = self._cue_detail_state(
                spike_aud_cue, self.cfg["dims"]["D_aud"], batch, device, dtype)
            img_dec_state = self._fuse_decoder_state(
                mem["v_img_from_A"], img_detail, "img")
            aud_dec_state = self._fuse_decoder_state(
                mem["v_aud_from_A"], aud_detail, "aud")
        elif self.detach_value_for_recon:
            img_dec_state = img_dec_state.detach()
            aud_dec_state = aud_dec_state.detach()

        out["img_detail_state"] = img_detail
        out["aud_detail_state"] = aud_detail
        out["recovered_img"] = self.image_decoder(img_dec_state)

        coarse_aud = self.audio_decoder(aud_dec_state)
        out["recovered_aud_coarse"] = coarse_aud
        recovered_aud = coarse_aud
        if (self.audio_refiner is not None
                and x_aud_cue is not None
                and aud_cue_mask is not None):
            mask = aud_cue_mask.to(device=coarse_aud.device, dtype=coarse_aud.dtype)
            delta = self.audio_refiner(coarse_aud, x_aud_cue, mask)
            recovered_aud = (coarse_aud + mask * delta).clamp(0.0, 1.0)
        out["recovered_aud"] = recovered_aud
        return out

    @torch.no_grad()
    def infer(self, x_img_cue=None, x_aud_cue=None, aud_cue_mask=None):
        self.eval()
        return self.forward(x_img_cue=x_img_cue, x_aud_cue=x_aud_cue,
                            training_mode=False, phase="readout",
                            aud_cue_mask=aud_cue_mask)
