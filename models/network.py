"""顶层跨模态 SNN 联想记忆网络（统一 cue->补全 接口）。"""

import torch
import torch.nn as nn

from .encoders import ImageSNNEncoder, AudioSNNEncoder
from .memory import CrossModalAttractorMemory
from .decoders import (ClassifierHead, ImageDecoder, ImageRefiner,
                       AudioDecoder, AudioRefiner)
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

        cross_cfg = cfg.get("cross_key_conditioning", {})
        self.use_cross_key_conditioning = bool(cross_cfg.get("enabled", False))
        self.build_cross_key_conditioning = (
            self.use_cross_key_conditioning
            or bool(cross_cfg.get("build_modules", False)))
        self.cross_key_detach = bool(cross_cfg.get("detach_key", True))
        if self.build_cross_key_conditioning:
            self.aud_to_img_cross_proj = nn.Linear(
                d["N_key_aud"], d["N_value_img"])
            self.img_to_aud_cross_proj = nn.Linear(
                d["N_key_img"], d["N_value_aud"])
            self.aud_to_img_cross_gate = nn.Linear(
                d["N_value_img"] + d["N_key_aud"], 1)
            self.img_to_aud_cross_gate = nn.Linear(
                d["N_value_aud"] + d["N_key_img"], 1)
            for projector in (self.aud_to_img_cross_proj,
                              self.img_to_aud_cross_proj):
                nn.init.zeros_(projector.weight)
                nn.init.zeros_(projector.bias)
            nn.init.zeros_(self.aud_to_img_cross_gate.bias)
            nn.init.zeros_(self.img_to_aud_cross_gate.bias)
        else:
            self.aud_to_img_cross_proj = None
            self.img_to_aud_cross_proj = None
            self.aud_to_img_cross_gate = None
            self.img_to_aud_cross_gate = None

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
        img_refiner_cfg = cfg.get("image_refiner", {})
        self.use_image_refiner = img_refiner_cfg.get("enabled", False)
        self.image_refiner_pasteback_only = img_refiner_cfg.get(
            "pasteback_only", False)
        self.image_refiner_visible_paste_back = img_refiner_cfg.get(
            "visible_paste_back", False)
        if self.use_image_refiner:
            self.image_refiner = ImageRefiner(
                hidden_ch=int(img_refiner_cfg.get("hidden_ch", 32)),
                blocks=int(img_refiner_cfg.get("blocks", 3)),
                delta_scale=float(img_refiner_cfg.get("delta_scale", 1.0)),
                max_dilation=int(img_refiner_cfg.get("max_dilation", 4)))
        else:
            self.image_refiner = None

        aud_dec_ch = s.get("aud_decoder_base_ch", 128)
        self.audio_decoder = AudioDecoder(
            aud_decoder_in, ac["n_mels"], ac["n_frames"],
            base_ch=aud_dec_ch,
            start_hw=s.get("aud_decoder_start_hw", 4),
            refine_blocks=s.get("aud_decoder_refine_blocks", 0),
            refine_type=s.get("aud_refine_type", "plain"))

        refiner_cfg = cfg.get("audio_refiner", {})
        self.use_audio_refiner = refiner_cfg.get("enabled", False)
        self.audio_refiner_bypass = bool(refiner_cfg.get("bypass", False))
        self.audio_refiner_pasteback_only = refiner_cfg.get(
            "pasteback_only", False)
        self.refiner_visible_paste_back = refiner_cfg.get(
            "visible_paste_back", False)
        if self.use_audio_refiner:
            self.audio_refiner = AudioRefiner(
                ac["n_mels"], ac["n_frames"],
                hidden_ch=int(refiner_cfg.get("hidden_ch", 32)),
                blocks=int(refiner_cfg.get("blocks", 2)),
                delta_scale=float(refiner_cfg.get("delta_scale", 1.0)),
                max_dilation=int(refiner_cfg.get("max_dilation", 16)))
        else:
            self.audio_refiner = None
        if self.audio_refiner is not None and self.audio_refiner_bypass:
            for param in self.audio_refiner.parameters():
                param.requires_grad_(False)

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

    def _cross_key_residual(self, base_value, cross_key_rate, modality,
                            disabled=False):
        batch = base_value.size(0)
        zeros = base_value.new_zeros(batch)
        stats = {
            "gate": None,
            "residual_norm": zeros,
            "value_norm": base_value.norm(dim=1),
            "ratio": zeros,
        }
        if (disabled or not self.use_cross_key_conditioning
                or cross_key_rate is None):
            return torch.zeros_like(base_value), stats

        if cross_key_rate.dim() != 2 or cross_key_rate.size(0) != batch:
            raise ValueError(
                "cross_key_rate must have shape [B,D], got "
                f"{tuple(cross_key_rate.shape)} for batch={batch}")
        cross_key_rate = cross_key_rate.to(
            device=base_value.device, dtype=base_value.dtype)
        if self.cross_key_detach:
            cross_key_rate = cross_key_rate.detach()

        if modality == "img":
            projector = self.aud_to_img_cross_proj
            gate_layer = self.aud_to_img_cross_gate
        elif modality == "aud":
            projector = self.img_to_aud_cross_proj
            gate_layer = self.img_to_aud_cross_gate
        else:
            raise ValueError(f"Unknown modality: {modality}")

        projected = projector(cross_key_rate)
        gate = torch.sigmoid(
            gate_layer(torch.cat([base_value, cross_key_rate], dim=1)))
        residual = gate * projected
        residual_norm = residual.norm(dim=1)
        value_norm = base_value.norm(dim=1)
        stats = {
            "gate": gate,
            "residual_norm": residual_norm,
            "value_norm": value_norm,
            "ratio": residual_norm / value_norm.clamp_min(1e-8),
        }
        return residual, stats

    def _fuse_decoder_state(self, value_state, raw_detail, modality,
                            cross_key_rate=None, disable_cross_key=False,
                            return_cross_stats=False):
        base_value = (value_state.detach() if self.detach_value_for_recon
                      else value_state)
        cross_residual, cross_stats = self._cross_key_residual(
            base_value, cross_key_rate, modality, disabled=disable_cross_key)
        fused_value = base_value + cross_residual
        if not self.use_detail_conditioning:
            if return_cross_stats:
                return fused_value, cross_stats
            return fused_value

        if modality == "img":
            detail = self.img_detail_projector(raw_detail)
            if self.detail_fusion == "gated_concat":
                gate = self.img_detail_gate(
                    torch.cat([fused_value, detail], dim=1))
                detail = gate * detail
        elif modality == "aud":
            detail = self.aud_detail_projector(raw_detail)
            if self.detail_fusion == "gated_concat":
                gate = self.aud_detail_gate(
                    torch.cat([fused_value, detail], dim=1))
                detail = gate * detail
        else:
            raise ValueError(f"Unknown modality: {modality}")
        decoder_state = torch.cat([fused_value, detail], dim=1)
        if return_cross_stats:
            return decoder_state, cross_stats
        return decoder_state

    @staticmethod
    def _prob_to_logits(prob, eps=1e-4):
        prob = prob.clamp(eps, 1.0 - eps)
        return torch.logit(prob)

    def _apply_image_refiner(self, coarse_logits, img_cue, img_mask):
        if img_cue is None or img_mask is None:
            return coarse_logits
        mask = img_mask.to(device=coarse_logits.device,
                           dtype=coarse_logits.dtype)
        coarse_prob = torch.sigmoid(coarse_logits)
        if self.image_refiner is not None:
            delta = self.image_refiner(coarse_prob, img_cue, mask)
            if self.image_refiner_visible_paste_back:
                pred = (coarse_prob + delta).clamp(0.0, 1.0)
                final_prob = mask * pred + (1.0 - mask) * img_cue
            else:
                final_prob = (coarse_prob + mask * delta).clamp(0.0, 1.0)
        elif self.image_refiner_pasteback_only:
            final_prob = mask * coarse_prob + (1.0 - mask) * img_cue
        else:
            return coarse_logits
        return self._prob_to_logits(final_prob)

    def _finalize_audio(self, decoder_aud, aud_cue, aud_mask):
        """Return the single public audio reconstruction.

        v11c bypasses the legacy AudioRefiner, so its decoder prediction is the
        final result.  The refiner module is still constructed when configured
        because parent checkpoints contain its parameters and must load
        strictly.
        """
        if self.audio_refiner_bypass:
            return decoder_aud
        if aud_cue is None or aud_mask is None:
            return decoder_aud
        mask = aud_mask.to(device=decoder_aud.device, dtype=decoder_aud.dtype)
        if self.audio_refiner is not None:
            delta = self.audio_refiner(decoder_aud, aud_cue, mask)
            if self.refiner_visible_paste_back:
                pred = (decoder_aud + delta).clamp(0.0, 1.0)
                return mask * pred + (1.0 - mask) * aud_cue
            return (decoder_aud + mask * delta).clamp(0.0, 1.0)
        if self.audio_refiner_pasteback_only:
            return mask * decoder_aud + (1.0 - mask) * aud_cue
        return decoder_aud

    def forward(self, x_img_cue=None, x_aud_cue=None,
                x_img_target=None, x_aud_target=None,
                training_mode=False, phase="readout",
                img_cue_mask=None, aud_cue_mask=None,
                cross_key_img_rate_override=None,
                cross_key_aud_rate_override=None,
                disable_img_to_aud_cross=False,
                disable_aud_to_img_cross=False):
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

        img_key_rate = cross_key_img_rate_override
        if img_key_rate is None and mem.get("key_img") is not None:
            img_key_rate = rate(mem["key_img"])
        aud_key_rate = cross_key_aud_rate_override
        if aud_key_rate is None and mem.get("key_aud") is not None:
            aud_key_rate = rate(mem["key_aud"])

        img_dec_state, aud_to_img_stats = self._fuse_decoder_state(
            mem["v_img_from_A"], img_detail, "img",
            cross_key_rate=aud_key_rate,
            disable_cross_key=disable_aud_to_img_cross,
            return_cross_stats=True)
        aud_dec_state, img_to_aud_stats = self._fuse_decoder_state(
            mem["v_aud_from_A"], aud_detail, "aud",
            cross_key_rate=img_key_rate,
            disable_cross_key=disable_img_to_aud_cross,
            return_cross_stats=True)

        out["img_detail_state"] = img_detail
        out["aud_detail_state"] = aud_detail
        out["aud_to_img_cross_gate"] = aud_to_img_stats["gate"]
        out["aud_to_img_cross_residual_norm"] = aud_to_img_stats["residual_norm"]
        out["aud_to_img_cross_value_norm"] = aud_to_img_stats["value_norm"]
        out["aud_to_img_cross_ratio"] = aud_to_img_stats["ratio"]
        out["img_to_aud_cross_gate"] = img_to_aud_stats["gate"]
        out["img_to_aud_cross_residual_norm"] = img_to_aud_stats["residual_norm"]
        out["img_to_aud_cross_value_norm"] = img_to_aud_stats["value_norm"]
        out["img_to_aud_cross_ratio"] = img_to_aud_stats["ratio"]
        coarse_img = self.image_decoder(img_dec_state)
        out["recovered_img_coarse"] = coarse_img
        out["recovered_img"] = self._apply_image_refiner(
            coarse_img, x_img_cue, img_cue_mask)

        decoder_aud = self.audio_decoder(aud_dec_state)
        out["recovered_aud"] = self._finalize_audio(
            decoder_aud, x_aud_cue, aud_cue_mask)
        return out

    @torch.no_grad()
    def infer(self, x_img_cue=None, x_aud_cue=None,
              img_cue_mask=None, aud_cue_mask=None,
              cross_key_img_rate_override=None,
              cross_key_aud_rate_override=None,
              disable_img_to_aud_cross=False,
              disable_aud_to_img_cross=False):
        self.eval()
        return self.forward(x_img_cue=x_img_cue, x_aud_cue=x_aud_cue,
                            training_mode=False, phase="readout",
                            img_cue_mask=img_cue_mask,
                            aud_cue_mask=aud_cue_mask,
                            cross_key_img_rate_override=(
                                cross_key_img_rate_override),
                            cross_key_aud_rate_override=(
                                cross_key_aud_rate_override),
                            disable_img_to_aud_cross=disable_img_to_aud_cross,
                            disable_aud_to_img_cross=disable_aud_to_img_cross)
