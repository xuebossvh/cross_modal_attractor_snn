"""Provenance and strict loading for frozen-parent adapter experiments."""

import hashlib

import torch

ADAPTER_PREFIXES = ("img_cross_adapter.", "aud_cross_adapter.")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def base_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith(ADAPTER_PREFIXES):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_parent(model, state, path):
    expected = model.cfg["train"].get("parent_sha256")
    if not expected or file_sha256(path) != expected:
        raise RuntimeError("Frozen parent SHA256 mismatch (or missing parent_sha256)")
    parent_cfg = state.get("cfg")
    if not parent_cfg or parent_cfg.get("cross_key_conditioning", {}).get("enabled"):
        raise RuntimeError("Expected a configured, Cross-Key-disabled control parent")
    for section in ("snn", "dims", "index", "value", "ablation", "detail_conditioning",
                    "image_refiner", "audio_refiner", "audio"):
        if parent_cfg.get(section) != model.cfg.get(section):
            raise RuntimeError(f"Frozen parent forward configuration mismatch: {section}")
    model.parent_sha256 = expected
    model.parent_audio_norm_stats = parent_cfg.get("_audio_norm_stats")


def load_frozen_parent(model, state, path):
    validate_parent(model, state, path)
    result = model.load_state_dict(state["model"], strict=False)
    bad = [key for key in result.missing_keys if not key.startswith(ADAPTER_PREFIXES)]
    if bad or result.unexpected_keys:
        raise RuntimeError(f"Incompatible frozen parent: {bad}, {result.unexpected_keys}")
    model.frozen_base_digest = base_digest(model)


def frozen_metadata(model):
    if not model.freeze_base:
        return {}
    current = base_digest(model)
    if current != getattr(model, "frozen_base_digest", None):
        raise RuntimeError("Frozen base parameters or buffers changed")
    return {"parent_sha256": model.parent_sha256, "base_sha256": current}


def verify_frozen_resume(model, state):
    metadata = state.get("frozen_base", {})
    expected = model.cfg["train"].get("parent_sha256")
    if not expected or metadata.get("parent_sha256") != expected:
        raise RuntimeError("Resume checkpoint has no matching frozen-parent provenance")
    saved_cfg = state.get("cfg", {})
    for section in ("cross_key_conditioning", "snn", "dims", "index", "value", "ablation",
                    "detail_conditioning", "image_refiner", "audio_refiner", "audio"):
        if saved_cfg.get(section) != model.cfg.get(section):
            raise RuntimeError(f"Resume configuration mismatch: {section}")
    model.parent_sha256 = expected
    model.parent_audio_norm_stats = saved_cfg.get("_audio_norm_stats")
    model.frozen_base_digest = metadata.get("base_sha256")
    frozen_metadata(model)


def verify_audio_normalization(model, cfg):
    if not model.freeze_base:
        return
    expected = getattr(model, "parent_audio_norm_stats", None)
    actual = cfg.get("_audio_norm_stats")
    if expected != actual:
        raise RuntimeError("Frozen parent audio normalization statistics do not match dataset")


def load_evaluation_checkpoint(model, path, device):
    state = torch.load(path, map_location=device)
    if model.freeze_base and model.cfg["train"].get("evaluation_only", False):
        load_frozen_parent(model, state, path)
    else:
        model.load_state_dict(state["model"], strict=True)
        if model.freeze_base:
            verify_frozen_resume(model, state)
    model.eval()
    return state
