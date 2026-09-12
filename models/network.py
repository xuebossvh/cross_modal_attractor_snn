"""顶层跨模态 SNN 联想记忆网络（统一 cue->补全 接口）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ImageSNNEncoder, AudioSNNEncoder
from .memory import CrossModalAttractorMemory
from .decoders import (ClassifierHead, ImageDecoder, ImageRefiner,
                       AudioDecoder, AudioRefiner, AudioLocalCueProjector,
                       MaskedCrossKeyAdapter)
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
        self.cross_key_mode = cross_cfg.get("mode", "value_residual")
        if self.cross_key_mode not in ("value_residual", "masked_feature"):
            raise ValueError(f"Unknown Cross-Key mode: {self.cross_key_mode}")
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

        cross_detail_cfg = cfg.get("cross_detail_conditioning", {}) or {}
        self.use_cross_detail_conditioning = bool(
            cross_detail_cfg.get("enabled", False))
        self.build_cross_detail_conditioning = (
            self.use_cross_detail_conditioning
            or bool(cross_detail_cfg.get("build_modules", False)))
        self.cross_detail_detach = bool(
            cross_detail_cfg.get("detach_source", True))
        if self.use_cross_detail_conditioning and not self.use_detail_conditioning:
            raise ValueError(
                "cross_detail_conditioning requires detail_conditioning.enabled=true")
        if self.build_cross_detail_conditioning:
            # Cross-Detail 使用 Key 前一层的高维实例特征，并加到目标模态
            # 已有 detail channel；因此不改变 v11c Decoder 输入尺寸。
            self.aud_to_img_detail_proj = nn.Linear(
                d["aud_hidden"], self.img_detail_dim)
            self.img_to_aud_detail_proj = nn.Linear(
                d["img_hidden"], self.aud_detail_dim)
            self.aud_to_img_detail_gate = nn.Linear(
                d["N_value_img"] + self.img_detail_dim + 1,
                self.img_detail_dim)
            self.img_to_aud_detail_gate = nn.Linear(
                d["N_value_aud"] + self.aud_detail_dim + 1,
                self.aud_detail_dim)
            for projector in (self.aud_to_img_detail_proj,
                              self.img_to_aud_detail_proj):
                nn.init.zeros_(projector.weight)
                nn.init.zeros_(projector.bias)
            nn.init.zeros_(self.aud_to_img_detail_gate.bias)
            nn.init.zeros_(self.img_to_aud_detail_gate.bias)
        else:
            self.aud_to_img_detail_proj = None
            self.img_to_aud_detail_proj = None
            self.aud_to_img_detail_gate = None
            self.img_to_aud_detail_gate = None

        pair_cfg = cfg.get("pair_alignment", {}) or {}
        self.use_pair_alignment = bool(pair_cfg.get("enabled", False))
        self.build_pair_alignment = (
            self.use_pair_alignment or bool(pair_cfg.get("build_modules", False)))
        self.pair_embed_dim = int(pair_cfg.get("embed_dim", 128))
        if self.build_pair_alignment:
            self.img_pair_projector = nn.Sequential(
                nn.Linear(d["img_hidden"], self.pair_embed_dim),
                nn.LayerNorm(self.pair_embed_dim),
            )
            self.aud_pair_projector = nn.Sequential(
                nn.Linear(d["aud_hidden"], self.pair_embed_dim),
                nn.LayerNorm(self.pair_embed_dim),
            )
        else:
            self.img_pair_projector = None
            self.aud_pair_projector = None

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
        local_cfg = cfg.get("audio_local_cue", {}) or {}
        self.use_audio_local_cue = bool(local_cfg.get("enabled", False))
        if self.use_audio_local_cue:
            self.audio_local_cue_projector = AudioLocalCueProjector(
                int(s.get("aud_conv_ch2", 32)),
                self.audio_decoder.feature_channels,
                int(local_cfg.get("hidden_channels", 32)))
        else:
            self.audio_local_cue_projector = None

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

        self.freeze_base = bool(cfg.get("train", {}).get("freeze_base", False))
        if self.freeze_base:
            if self.cross_key_mode != "masked_feature":
                raise ValueError("freeze_base requires masked_feature Cross-Key")
            if self.use_cross_detail_conditioning or self.use_pair_alignment:
                raise ValueError("Frozen category binding excludes Cross-Detail/pair loss")
            for param in self.parameters():
                param.requires_grad_(False)
        # Construct after every parent module so seeded parent initialization is unchanged.
        if (self.cross_key_mode == "masked_feature"
                and cross_cfg.get("build_feature_modules", True)):
            hidden = int(cross_cfg.get("hidden_ch", 32))
            key_ch = int(cross_cfg.get("key_channels", 16))
            self.img_cross_adapter = MaskedCrossKeyAdapter(
                32, d["N_key_aud"], hidden, key_ch)
            self.aud_cross_adapter = MaskedCrossKeyAdapter(
                self.audio_decoder.feature_channels, d["N_key_img"], hidden, key_ch)
        else:
            self.img_cross_adapter = None
            self.aud_cross_adapter = None
        if self.freeze_base:
            self.train(False)

    def train(self, mode=True):
        if getattr(self, "freeze_base", False):
            super().train(False)
            for adapter in (self.img_cross_adapter, self.aud_cross_adapter):
                if adapter is not None:
                    adapter.train(mode)
            return self
        return super().train(mode)

    @torch.no_grad()
    def classify_recovered(self, image=None, audio=None):
        """Internal content consistency, not an independent recognition oracle."""
        img_spikes = self.img_encoder(image) if image is not None else None
        aud_spikes = (self.aud_encoder(self._normalize_audio_for_encoder(audio))
                      if audio is not None else None)
        mem = self.memory(spike_img_cue=img_spikes, spike_aud_cue=aud_spikes,
                          phase="readout")
        return self.classifier(mem["index_state"])

    def _decode_masked_cross(self, state, modality, key, cue, mask,
                             source_missing_ratio, disabled, local_cue=None):
        decoder = self.image_decoder if modality == "img" else self.audio_decoder
        adapter = self.img_cross_adapter if modality == "img" else self.aud_cross_adapter
        finalize = self._apply_image_refiner if modality == "img" else self._finalize_audio
        features = decoder.forward_features(state)
        if (modality == "aud" and self.use_audio_local_cue
                and self.audio_local_cue_projector is not None
                and local_cue is not None):
            local_cue = local_cue.detach()
            features = features + self.audio_local_cue_projector(
                local_cue, features.shape[-2:])
        coarse = decoder.decode_features(features)
        base = finalize(coarse, cue, mask)
        if disabled or not self.use_cross_key_conditioning or key is None or adapter is None:
            return base, coarse, None
        region = torch.ones_like(base) if cue is None else (
            torch.zeros_like(base) if mask is None else mask.to(base))
        if not region.any():
            return base, coarse, None
        cue_map = torch.zeros_like(base) if cue is None else cue.to(base)
        if region.dim() == 3:
            region, cue_map = region.unsqueeze(1), cue_map.unsqueeze(1)
        if region.shape[-2:] != features.shape[-2:]:
            raise ValueError("masked_feature requires decoder features at target resolution")
        updated, stats = adapter(features, key, cue_map, region, source_missing_ratio)
        corrected_coarse = decoder.decode_features(updated)
        corrected = finalize(corrected_coarse, cue, mask)
        output_mask = region if base.dim() == 4 else region.squeeze(1)
        # Protect visible baseline pixels after both the head and refiner receptive fields.
        corrected = torch.where(output_mask.bool(), corrected, base)
        corrected_coarse = torch.where(output_mask.bool(), corrected_coarse, coarse)
        return corrected, corrected_coarse, stats

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

    def _instance_detail_state(self, spikes):
        """Cross-Detail 源：Key 前一层脉冲率；模态缺失时保持 None。"""
        if spikes is None:
            return None
        detail = rate(spikes)
        if self.cross_detail_detach:
            detail = detail.detach()
        return detail

    def _pair_embedding(self, detail, modality):
        """Project a pre-Key instance state into the shared pair space."""
        if detail is None or not self.build_pair_alignment:
            return None
        projector = (self.img_pair_projector if modality == "img"
                     else self.aud_pair_projector)
        return F.normalize(projector(detail), dim=1)

    @staticmethod
    def _missing_ratio(cue, mask, batch, device, dtype):
        """目标模态缺失比例：模态缺席=1，干净无 mask=0。"""
        if cue is None:
            return torch.ones(batch, 1, device=device, dtype=dtype)
        if mask is None:
            return torch.zeros(batch, 1, device=device, dtype=dtype)
        return mask.to(device=device, dtype=dtype).flatten(1).mean(
            dim=1, keepdim=True)

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
        if (disabled or self.cross_key_mode == "masked_feature"
                or not self.use_cross_key_conditioning
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

    def _cross_detail_residual(self, base_value, cross_detail_rate, modality,
                               target_missing_ratio=None, disabled=False):
        """把对侧 Key 前实例特征投影成目标模态的逐维 detail residual。"""
        batch = base_value.size(0)
        detail_dim = (self.img_detail_dim if modality == "img"
                      else self.aud_detail_dim)
        zeros = base_value.new_zeros(batch)
        empty = base_value.new_zeros(batch, detail_dim)
        stats = {
            "detail_gate": None,
            "detail_residual_norm": zeros,
            "detail_ratio": zeros,
        }
        if (disabled or not self.use_cross_detail_conditioning
                or cross_detail_rate is None):
            return empty, stats

        if cross_detail_rate.dim() != 2 or cross_detail_rate.size(0) != batch:
            raise ValueError(
                "cross_detail_rate must have shape [B,D], got "
                f"{tuple(cross_detail_rate.shape)} for batch={batch}")
        cross_detail_rate = cross_detail_rate.to(
            device=base_value.device, dtype=base_value.dtype)
        if self.cross_detail_detach:
            cross_detail_rate = cross_detail_rate.detach()

        if modality == "img":
            projector = self.aud_to_img_detail_proj
            gate_layer = self.aud_to_img_detail_gate
        elif modality == "aud":
            projector = self.img_to_aud_detail_proj
            gate_layer = self.img_to_aud_detail_gate
        else:
            raise ValueError(f"Unknown modality: {modality}")

        projected = projector(cross_detail_rate)
        if target_missing_ratio is None:
            target_missing_ratio = base_value.new_zeros(batch, 1)
        else:
            target_missing_ratio = target_missing_ratio.to(
                device=base_value.device, dtype=base_value.dtype)
            if target_missing_ratio.dim() == 1:
                target_missing_ratio = target_missing_ratio.unsqueeze(1)
            if target_missing_ratio.shape != (batch, 1):
                raise ValueError(
                    "target_missing_ratio must have shape [B,1], got "
                    f"{tuple(target_missing_ratio.shape)}")
        gate = torch.sigmoid(
            gate_layer(torch.cat(
                [base_value, projected, target_missing_ratio], dim=1)))
        residual = gate * projected
        residual_norm = residual.norm(dim=1)
        return residual, {
            "detail_gate": gate,
            "detail_residual_norm": residual_norm,
            "detail_ratio": residual_norm / base_value.norm(
                dim=1).clamp_min(1e-8),
        }

    def _fuse_decoder_state(self, value_state, raw_detail, modality,
                            cross_key_rate=None, disable_cross_key=False,
                            raw_cross_detail=None,
                            target_missing_ratio=None,
                            disable_cross_detail=False,
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
        cross_detail, detail_stats = self._cross_detail_residual(
            fused_value, raw_cross_detail, modality,
            target_missing_ratio=target_missing_ratio,
            disabled=disable_cross_detail)
        detail = detail + cross_detail
        cross_stats.update(detail_stats)
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
            if (self.refiner_visible_paste_back
                    and aud_cue is not None and aud_mask is not None):
                mask = aud_mask.to(
                    device=decoder_aud.device, dtype=decoder_aud.dtype)
                return mask * decoder_aud + (1.0 - mask) * aud_cue
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
                cross_detail_img_rate_override=None,
                cross_detail_aud_rate_override=None,
                disable_img_to_aud_cross=False,
                disable_aud_to_img_cross=False,
                disable_img_to_aud_detail=False,
                disable_aud_to_img_detail=False):
        assert (x_img_cue is not None) or (x_aud_cue is not None), \
            "至少需要一种 cue 模态作为输入"

        if not training_mode:
            phase = "readout"
            x_img_target = None
            x_aud_target = None

        spike_img_cue = None
        spike_img_instance = None
        if x_img_cue is not None:
            spike_img_cue, spike_img_instance = (
                self.img_encoder.forward_with_detail(x_img_cue))
        spike_aud_cue = None
        spike_aud_instance = None
        spike_aud_local = None
        if x_aud_cue is not None:
            spike_aud_cue, spike_aud_instance, spike_aud_local = (
                self.aud_encoder.forward_with_local(
                    self._normalize_audio_for_encoder(x_aud_cue)))

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
        aud_local_cue = None
        if self.use_detail_conditioning:
            batch = mem["index_state"].size(0)
            device = mem["index_state"].device
            dtype = mem["index_state"].dtype
            img_detail = self._cue_detail_state(
                spike_img_cue, self.cfg["dims"]["D_img"], batch, device, dtype)
            aud_detail = self._cue_detail_state(
                spike_aud_cue, self.cfg["dims"]["D_aud"], batch, device, dtype)
            if self.use_audio_local_cue and spike_aud_local is not None:
                aud_local_cue = rate(spike_aud_local)

        img_key_rate = cross_key_img_rate_override
        if img_key_rate is None and mem.get("key_img") is not None:
            img_key_rate = rate(mem["key_img"])
        aud_key_rate = cross_key_aud_rate_override
        if aud_key_rate is None and mem.get("key_aud") is not None:
            aud_key_rate = rate(mem["key_aud"])

        img_cross_detail = cross_detail_img_rate_override
        if img_cross_detail is None:
            img_cross_detail = self._instance_detail_state(spike_img_instance)
        aud_cross_detail = cross_detail_aud_rate_override
        if aud_cross_detail is None:
            aud_cross_detail = self._instance_detail_state(spike_aud_instance)

        batch = mem["index_state"].size(0)
        device = mem["index_state"].device
        dtype = mem["index_state"].dtype
        img_missing_ratio = self._missing_ratio(
            x_img_cue, img_cue_mask, batch, device, dtype)
        aud_missing_ratio = self._missing_ratio(
            x_aud_cue, aud_cue_mask, batch, device, dtype)

        img_dec_state, aud_to_img_stats = self._fuse_decoder_state(
            mem["v_img_from_A"], img_detail, "img",
            cross_key_rate=aud_key_rate,
            disable_cross_key=disable_aud_to_img_cross,
            raw_cross_detail=aud_cross_detail,
            target_missing_ratio=img_missing_ratio,
            disable_cross_detail=disable_aud_to_img_detail,
            return_cross_stats=True)
        aud_dec_state, img_to_aud_stats = self._fuse_decoder_state(
            mem["v_aud_from_A"], aud_detail, "aud",
            cross_key_rate=img_key_rate,
            disable_cross_key=disable_img_to_aud_cross,
            raw_cross_detail=img_cross_detail,
            target_missing_ratio=aud_missing_ratio,
            disable_cross_detail=disable_img_to_aud_detail,
            return_cross_stats=True)

        out["img_detail_state"] = img_detail
        out["aud_detail_state"] = aud_detail
        out["img_cross_detail_state"] = img_cross_detail
        out["aud_cross_detail_state"] = aud_cross_detail
        out["img_pair_embedding"] = self._pair_embedding(
            img_cross_detail, "img")
        out["aud_pair_embedding"] = self._pair_embedding(
            aud_cross_detail, "aud")
        out["aud_to_img_cross_gate"] = aud_to_img_stats["gate"]
        out["aud_to_img_cross_residual_norm"] = aud_to_img_stats["residual_norm"]
        out["aud_to_img_cross_value_norm"] = aud_to_img_stats["value_norm"]
        out["aud_to_img_cross_ratio"] = aud_to_img_stats["ratio"]
        out["img_to_aud_cross_gate"] = img_to_aud_stats["gate"]
        out["img_to_aud_cross_residual_norm"] = img_to_aud_stats["residual_norm"]
        out["img_to_aud_cross_value_norm"] = img_to_aud_stats["value_norm"]
        out["img_to_aud_cross_ratio"] = img_to_aud_stats["ratio"]
        out["aud_to_img_detail_gate"] = aud_to_img_stats["detail_gate"]
        out["aud_to_img_detail_residual_norm"] = (
            aud_to_img_stats["detail_residual_norm"])
        out["aud_to_img_detail_ratio"] = aud_to_img_stats["detail_ratio"]
        out["img_to_aud_detail_gate"] = img_to_aud_stats["detail_gate"]
        out["img_to_aud_detail_residual_norm"] = (
            img_to_aud_stats["detail_residual_norm"])
        out["img_to_aud_detail_ratio"] = img_to_aud_stats["detail_ratio"]
        if self.cross_key_mode == "masked_feature":
            rec_img, coarse_img, img_stats = self._decode_masked_cross(
                img_dec_state, "img", aud_key_rate if x_aud_cue is not None else None,
                x_img_cue, img_cue_mask, aud_missing_ratio, disable_aud_to_img_cross)
            rec_aud, _, aud_stats = self._decode_masked_cross(
                aud_dec_state, "aud", img_key_rate if x_img_cue is not None else None,
                x_aud_cue, aud_cue_mask, img_missing_ratio, disable_img_to_aud_cross,
                local_cue=aud_local_cue)
            out["recovered_img_coarse"] = coarse_img
            out["recovered_img"], out["recovered_aud"] = rec_img, rec_aud
            for direction, stats in (("aud_to_img", img_stats), ("img_to_aud", aud_stats)):
                if stats is not None:
                    for key, value in stats.items():
                        out[f"{direction}_cross_{key}"] = value
            return out

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
              cross_detail_img_rate_override=None,
              cross_detail_aud_rate_override=None,
              disable_img_to_aud_cross=False,
              disable_aud_to_img_cross=False,
              disable_img_to_aud_detail=False,
              disable_aud_to_img_detail=False):
        self.eval()
        return self.forward(x_img_cue=x_img_cue, x_aud_cue=x_aud_cue,
                            training_mode=False, phase="readout",
                            img_cue_mask=img_cue_mask,
                            aud_cue_mask=aud_cue_mask,
                            cross_key_img_rate_override=(
                                cross_key_img_rate_override),
                            cross_key_aud_rate_override=(
                                cross_key_aud_rate_override),
                            cross_detail_img_rate_override=(
                                cross_detail_img_rate_override),
                            cross_detail_aud_rate_override=(
                                cross_detail_aud_rate_override),
                            disable_img_to_aud_cross=disable_img_to_aud_cross,
                            disable_aud_to_img_cross=disable_aud_to_img_cross,
                            disable_img_to_aud_detail=disable_img_to_aud_detail,
                            disable_aud_to_img_detail=disable_aud_to_img_detail)
