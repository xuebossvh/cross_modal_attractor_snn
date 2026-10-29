# CBDNet FCN provenance

The local contamination detector in [noise_detector.py](../../models/noise_detector.py)
adapts the noise-estimation FCN from the MIT-licensed, unofficial PyTorch
implementation of CBDNet by IDKiro:

- Repository: <https://github.com/IDKiro/CBDNet-pytorch>
- Inspected commit: `09a2e55b2098039ee99ada8c634a06fc28c6d8a1`.
- Source: [model/cbdnet.py, class FCN](https://github.com/IDKiro/CBDNet-pytorch/blob/09a2e55b2098039ee99ada8c634a06fc28c6d8a1/model/cbdnet.py#L48).
- Upstream license: [MIT](https://github.com/IDKiro/CBDNet-pytorch/blob/09a2e55b2098039ee99ada8c634a06fc28c6d8a1/LICENSE).
  The full copyright and permission notice is retained in [LICENSE](LICENSE).
- Paper: Guo et al., *Toward Convolutional Blind Denoising of Real Photographs*,
  CVPR 2019, <https://arxiv.org/abs/1807.04686>.

## Adaptation

Upstream's FCN estimates three-channel RGB noise levels with five 3x3
convolutions and ReLU activations. Here the input and output are single-channel,
and the last ReLU is removed to produce contamination logits. The four hidden
layers default to 32 channels; no pooling changes the spatial resolution.
Weighted binary cross entropy supervises synthetic local contamination support.
The resulting hard mask selects model prediction at contaminated positions and
input observations elsewhere. Images and log-mel cues use separately trained
detectors.

This is a new task adaptation, not an original CBDNet binary-mask model or a
claim of its published denoising performance. No upstream pretrained weights,
denoising U-Net, dataset, or training pipeline are bundled. Detectors must be
trained on this project's local corruption distributions. Ground-truth masks
are training/evaluation labels and are not detector inputs at blind inference.

The official [IDCNN implementation](https://github.com/k-radlak/IDCNN) was also
examined because its task is impulse detection. Its GPL-3.0 source and weights
were not copied into this project.
