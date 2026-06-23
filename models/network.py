"""顶层跨模态 SNN 联想记忆网络（统一 cue->补全 接口）。"""

import torch
import torch.nn as nn

from .encoders import ImageSNNEncoder, AudioSNNEncoder
from .memory import CrossModalAttractorMemory
from .decoders import ClassifierHead, ImageDecoder, AudioDecoder
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
            base_ch=aud_dec_ch)

        self.use_audio_aux = ab.get("use_audio_aux_cls", True)
        if self.use_audio_aux:
            self.aux_aud_classifier = ClassifierHead(
                d["N_key_aud"], d["num_classes"])
        else:
            self.aux_aud_classifier = None

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
        spike_aud_cue = self.aud_encoder(x_aud_cue) if x_aud_cue is not None else None

        spike_img_tgt = None
        spike_aud_tgt = None
        if training_mode and phase == "binding":
            if x_img_target is not None:
                spike_img_tgt = self.img_encoder(x_img_target)
            if x_aud_target is not None:
                spike_aud_tgt = self.aud_encoder(x_aud_target)

        mem = self.memory(
            spike_img_cue=spike_img_cue, spike_aud_cue=spike_aud_cue,
            spike_img_target=spike_img_tgt, spike_aud_target=spike_aud_tgt,
            phase=phase)

        out = {
            "index_spikes": mem["index_spikes"],
            "index_state": mem["index_state"],
            "spike_img_cue": spike_img_cue, "spike_aud_cue": spike_aud_cue,
            "key_aud": mem.get("key_aud"),
            "v_img_from_A": mem["v_img_from_A"], "v_aud_from_A": mem["v_aud_from_A"],
            "v_img_target": mem["v_img_target"], "v_aud_target": mem["v_aud_target"],
        }

        out["logits"] = self.classifier(mem["index_state"])
        out["aux_aud_logits"] = None
        if (self.aux_aud_classifier is not None
                and mem.get("key_aud") is not None):
            out["aux_aud_logits"] = self.aux_aud_classifier(rate(mem["key_aud"]))

        out["recovered_img"] = self.image_decoder(mem["v_img_from_A"])
        out["recovered_aud"] = self.audio_decoder(mem["v_aud_from_A"])
        return out

    @torch.no_grad()
    def infer(self, x_img_cue=None, x_aud_cue=None):
        self.eval()
        return self.forward(x_img_cue=x_img_cue, x_aud_cue=x_aud_cue,
                            training_mode=False, phase="readout")
