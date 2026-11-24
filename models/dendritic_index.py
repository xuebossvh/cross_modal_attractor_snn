"""Two modality-specific dendritic compartments feeding a recurrent LIF soma."""

import math

import torch
from torch import nn

from .lif import rate, spike_fn


def _topk_mask(values, k):
    winners = values.topk(k, dim=-1).indices
    return torch.zeros_like(values).scatter_(-1, winners, 1)


class DendriticIndexLayer(nn.Module):
    """Recurrent Index with states reset at the start of each cue."""

    def __init__(self, n_key_img=128, n_key_aud=128, n_index=512, *,
                 dendritic_k=48, soma_k=96, dendritic_beta=0.8, soma_beta=0.9,
                 dendritic_reset=0.3, coincidence_scale=0.1, threshold=1.0,
                 surrogate_alpha=2.0, recurrent_scale=0.2):
        super().__init__()
        if not 1 <= dendritic_k <= n_index or not 1 <= soma_k <= n_index:
            raise ValueError("Competition sizes must be between 1 and n_index")
        if not all(0 <= value <= 1 for value in (dendritic_beta, soma_beta, dendritic_reset)):
            raise ValueError("Leak and reset coefficients must be in [0, 1]")
        if not math.isfinite(coincidence_scale) or coincidence_scale < 0:
            raise ValueError("Coincidence scale must be finite and nonnegative")

        self.image_projection = nn.Linear(n_key_img, n_index, bias=False)
        self.audio_projection = nn.Linear(n_key_aud, n_index, bias=False)
        self.recurrent = nn.Linear(n_index, n_index, bias=False)
        with torch.no_grad():
            self.recurrent.weight.mul_(recurrent_scale)
        self.n_index = n_index
        self.dendritic_k = dendritic_k
        self.soma_k = soma_k
        self.dendritic_beta = dendritic_beta
        self.soma_beta = soma_beta
        self.dendritic_reset = dendritic_reset
        self.coincidence_scale = coincidence_scale
        self.threshold = threshold
        self.surrogate_alpha = surrogate_alpha

    def forward(self, image_keys=None, audio_keys=None, *, cross_modal_current=None,
                return_details=True):
        """Read Key sequences [T, B, D], with None for an absent modality."""
        ref = image_keys if image_keys is not None else audio_keys
        if ref is None or ref.ndim != 3 or ref.size(0) == 0:
            raise ValueError("At least one nonempty Key sequence [T, B, D] is required")
        time_steps, batch = ref.shape[:2]
        for keys, projection in ((image_keys, self.image_projection),
                                 (audio_keys, self.audio_projection)):
            if keys is not None and keys.shape != (time_steps, batch, projection.in_features):
                raise ValueError("Key sequences must share T and B and match their projection")

        if cross_modal_current is not None and cross_modal_current.shape != (batch, self.n_index):
            raise ValueError("Cross-modal current must be [B, N_index]")
        shape = (time_steps, batch, self.n_index)
        image_current = self.image_projection(image_keys) if image_keys is not None else ref.new_zeros(shape)
        audio_current = self.audio_projection(audio_keys) if audio_keys is not None else ref.new_zeros(shape)
        image_voltage = ref.new_zeros((batch, self.n_index))
        audio_voltage = torch.zeros_like(image_voltage)
        soma_voltage = torch.zeros_like(image_voltage)
        soma_spikes = torch.zeros_like(image_voltage)
        traces = {key: [] for key in ("image_drive", "audio_drive", "soma_spikes")}
        if return_details:
            traces.update(image_voltage=[], audio_voltage=[], coincidence=[])

        for image_input, audio_input in zip(image_current, audio_current):
            image_voltage = self.dendritic_beta * image_voltage + image_input
            audio_voltage = self.dendritic_beta * audio_voltage + audio_input
            image_drive = image_voltage.relu().tanh()
            audio_drive = audio_voltage.relu().tanh()
            image_drive = image_drive * _topk_mask(image_drive, self.dendritic_k)
            audio_drive = audio_drive * _topk_mask(audio_drive, self.dendritic_k)
            coincidence = self.coincidence_scale * image_drive * audio_drive
            soma_voltage = (self.soma_beta * soma_voltage + image_drive + audio_drive
                            + coincidence + self.recurrent(soma_spikes))
            if cross_modal_current is not None:
                soma_voltage = soma_voltage + cross_modal_current
            soma_spikes = spike_fn(soma_voltage - self.threshold, self.surrogate_alpha)
            soma_spikes = soma_spikes * _topk_mask(soma_voltage, self.soma_k)

            if return_details:
                traces["image_voltage"].append(image_voltage)
                traces["audio_voltage"].append(audio_voltage)
                traces["coincidence"].append(coincidence)
            traces["image_drive"].append(image_drive)
            traces["audio_drive"].append(audio_drive)
            traces["soma_spikes"].append(soma_spikes)
            reset = soma_spikes.detach()
            soma_voltage = soma_voltage * (1 - reset)
            image_voltage = image_voltage * (1 - self.dendritic_reset * reset)
            audio_voltage = audio_voltage * (1 - self.dendritic_reset * reset)

        result = {key: torch.stack(sequence) for key, sequence in traces.items()}
        result["soma_rate"] = rate(result["soma_spikes"])
        # Dendrites contribute their mean drive rather than a spike count.
        result["classifier_state"] = torch.cat((result["image_drive"].mean(dim=0),
                                                result["audio_drive"].mean(dim=0),
                                                result["soma_rate"]), dim=-1)
        return result
