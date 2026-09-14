"""Development-only endpoint shared by SNN and recovery CNN model selection."""

import torch

from common import build_cue, resolve_train_corrupt_modes, set_seed, unpack_paired_batch


@torch.no_grad()
def validate_recovery(model, loader, cfg, device, cnn=False):
    from scripts.paper_baseline import region_mse

    was_training = model.training
    model.eval()
    protos = (loader.dataset.prototype_img.to(device), loader.dataset.prototype_aud.to(device))
    groups = {}
    devices = [device.index or 0] if device.type == "cuda" else []
    # Validation must not change the following epoch's random stream.
    import random
    import numpy as np
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            for bi, batch in enumerate(loader):
                img, aud, labels, _ = unpack_paired_batch(batch)
                img, aud, labels = img.to(device), aud.to(device), labels.to(device)
                family = resolve_train_corrupt_modes(cfg, 0, bi)
                for mi, mode in enumerate(("corrupt_aud_only", "clean_img_corrupt_aud", "corrupt_both")):
                    set_seed(20260915 + bi * 10 + mi)
                    ci, ca, masks = build_cue(img, aud, mode, cfg, severity=.4,
                        img_mode=family[0], aud_mode=family[1], return_masks=True)
                    if cnn:
                        logits, ri, ra = model(ci, ca, masks, *protos)
                    else:
                        out = model(x_img_cue=ci, x_aud_cue=ca, training_mode=False,
                                    img_cue_mask=masks["img"], aud_cue_mask=masks["aud"])
                        logits, ri, ra = out["logits"], out["recovered_img"].sigmoid(), out["recovered_aud"]
                    metrics = (float(region_mse(ra, aud, masks["aud"])),
                               float((ra-aud).square().mean()),
                               float((ri-img).square().mean()) if ci is not None else float((ri-protos[0][labels]).square().mean()),
                               float((logits.argmax(1) == labels).float().mean()))
                    key = (family, mode)
                    previous, count = groups.get(key, ([0.] * 4, 0))
                    groups[key] = ([a + b*len(labels) for a,b in zip(previous, metrics)], count+len(labels))
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        model.train(was_training)
    if not groups:
        raise ValueError("Empty validation split")
    means = [sum(values[j]/n for values,n in groups.values())/len(groups) for j in range(4)]
    return dict(score=means[0], aud_masked_mse=means[0], aud_mse=means[1], img_mse=means[2],
                acc=means[3], n=sum(n for _,n in groups.values()),
                pair_i2a_acc=float("nan"), pair_a2i_acc=float("nan"))
