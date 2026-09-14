"""Explicitly simple ANN baselines; not claimed to be parameter-matched SOTA."""

import torch
from torch import nn
from torch.nn import functional as F


class DigitCNN(nn.Module):
    def __init__(self, classes=10, width=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(2 * width, 4 * width, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)))
        self.head = nn.Linear(4 * width * 16, classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.head(self.features(x).flatten(1))


class CleanRecognizers(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.img = DigitCNN(classes)
        self.aud = DigitCNN(classes)


class MaskedConvDecoder(nn.Module):
    def __init__(self, classes=10, width=32):
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(2, width, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.ReLU())
        self.condition = nn.Linear(classes, 2 * width, bias=False)
        self.middle = nn.Sequential(
            nn.Conv2d(2 * width, 2 * width, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, 3, padding=4, dilation=4), nn.ReLU())
        self.up = nn.Sequential(nn.Conv2d(2 * width, width, 3, padding=1), nn.ReLU(),
                                nn.Conv2d(width, 1, 3, padding=1))

    def forward(self, cue, mask, condition):
        if cue.ndim == 3:
            cue = cue.unsqueeze(1)
        if mask is None:
            mask = torch.zeros_like(cue)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
        features = self.down(torch.cat((cue, mask), 1))
        features = features + self.condition(condition)[:, :, None, None]
        features = features + self.middle(features)
        return torch.sigmoid(self.up(F.interpolate(features, cue.shape[-2:], mode="nearest")))


def pasteback(pred, cue, mask):
    # Match the current SNN policy: a clean cue with mask=None is reconstructed.
    return pred if cue is None or mask is None else mask * pred + (1 - mask) * cue


class RecoveryCNN(nn.Module):
    def __init__(self, classes=10, conditioned=True):
        super().__init__()
        self.recognizers = CleanRecognizers(classes)
        self.img_decoder = MaskedConvDecoder(classes)
        self.aud_decoder = MaskedConvDecoder(classes)
        self.conditioned = conditioned

    def forward(self, img, aud, masks, proto_img, proto_aud):
        logits = []
        if img is not None:
            logits.append(self.recognizers.img(img))
        if aud is not None:
            logits.append(self.recognizers.aud(aud))
        fused = torch.stack(logits).mean(0)
        probs = fused.detach().softmax(1)
        condition = probs if self.conditioned else torch.zeros_like(probs)
        ri = torch.einsum("bc,cnhw->bnhw", probs, proto_img)
        ra = torch.einsum("bc,chw->bhw", probs, proto_aud)
        if img is not None:
            ri = self.img_decoder(img, masks.get("img"), condition)
        if aud is not None:
            ra = self.aud_decoder(aud, masks.get("aud"), condition).squeeze(1)
        return fused, pasteback(ri, img, masks.get("img")), pasteback(ra, aud, masks.get("aud"))
