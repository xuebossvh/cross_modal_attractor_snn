# v11e real-pair protocol

## Why the 300 FSDD recordings are not enough

v11d had 2,700 train and 300 test base recordings. Reusing a recording with a
new shift/noise seed changes its view, not its physical identity. It therefore
cannot create 60,000 genuine image-audio pairs, and exact-pair recovery is not
identifiable when an unrelated MNIST image is attached by digit class.

v11e changes the data domain to GRID audiovisual utterances. Each row contains
audio and video frames captured during one physical utterance. The image tensor
is a rank-pooled mouth-motion dynamic image computed from that row's frames;
the audio tensor is the log-mel feature from that row's WAV.

Official GRID archive: <https://zenodo.org/records/3625687>

## Split and counting rules

- One unique `source_id` is one unique pair, regardless of repeated training
  exposure or future augmentation.
- The manifest records `source_id`, `image_source_id`, and `audio_source_id`;
  all three must be identical.
- `pair_id`, image path, audio path, and source id must be globally unique.
- Speakers are disjoint across train, validation, and test.
- Validation/test features are never augmented.
- With the commonly available 33-speaker archive and the default 3/3 held-out
  speakers, the expected count is approximately 27k train, 3k validation, and
  3k test. `dataset_summary.json` is the authoritative count for a local copy.

`train_samples_per_epoch: 60000` matches MNIST's number of optimizer exposures
per epoch by sampling the real train set with replacement. It does not change
the unique-pair count. Results must report both quantities:

```text
unique train pairs: <manifest count>
optimizer exposures per epoch: 60000
```

For a size-controlled comparison, either subsample MNIST to the real GRID train
count or keep 60,000 exposures for both models. Never call repeated GRID rows
new real pairs.

## Preparation

Extract GRID audio and JPG frame archives first, then run:

```bash
python -u scripts/prepare_grid_v11e.py \
  --audio_root /root/autodl-tmp/grid/audio \
  --frames_root /root/autodl-tmp/grid/jpg \
  --output_root _data/grid_v11e
```

The script performs a training-only p1-p99 log-mel normalization fit, writes
precomputed 64x64 audio features and 28x28 motion images, and creates:

```text
_data/grid_v11e/pairs.csv
_data/grid_v11e/audio_norm_stats.pt
_data/grid_v11e/dataset_summary.json
```

Before training:

```bash
python -u scripts/smoke_test_v11e.py
python scripts/mkdir_outputs.py --config configs/v11e_control.yaml
python scripts/mkdir_outputs.py --config configs/v11e.yaml
```

## Main and control

- `configs/v11e.yaml`: Cross-Key + Cross-Detail + symmetric pair alignment.
  Same-digit different-source examples are hard negatives.
- `configs/v11e_control.yaml`: same data, split, architecture and Cross-Key;
  Cross-Detail injection and pair-alignment loss are disabled.
- Both run from scratch for 100 epochs because GRID is a different image/audio
  domain from MNIST/FSDD. A v11c or v11d checkpoint is not a valid initializer.
- Final checkpoints support resume; lowest validation composite-score weights
  are separately saved as `*_best.pt` and used by evaluation by default.

All v11e cue modes use the true counterpart from the same manifest row as the
target. Evaluation and demo output therefore say `paired-sample`, never
`category`. The main evaluation also reports exact-pair Recall@1 among
same-digit candidates in both image-to-audio and audio-to-image directions.
