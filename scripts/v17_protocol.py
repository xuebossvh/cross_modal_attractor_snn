"""Model arms shared by the v17 planner, training and exporter."""

ARMS = {
    "dendritic": (False, True, True),
    "dendritic_additive": (False, True, True),
    "shared_sum": (False, False, False),
    "three_state": (True, False, False),
    "mult": (False, True, False),
    "main_unidirectional": (True, True, False),
    "main": (True, True, False),
}


def configure_arm(cfg, arm):
    private, attention, dendritic = ARMS[arm]
    settings = cfg["v17"]
    settings["arm"] = arm
    settings["private_index"]["enabled"] = private
    settings["mult"]["enabled"] = attention
    if arm == "main":
        settings["mult"]["mode"] = "bidirectional"
    else:
        settings["mult"].pop("mode", None)
    dendrites = settings.setdefault("dendritic_index", {})
    dendrites["enabled"] = dendritic
    if arm == "dendritic_additive":
        dendrites["coincidence_scale"] = 0.0


def verify_arm(cfg):
    settings = cfg["v17"]
    arm = settings.get("arm")
    dendrites = settings.get("dendritic_index", {})
    active = (settings["private_index"]["enabled"], settings["mult"]["enabled"],
              dendrites.get("enabled", False))
    if ARMS.get(arm) != active:
        raise ValueError("v17 arm and module switches disagree")
    expected_attention = "bidirectional" if arm == "main" else "shared"
    if settings["mult"].get("mode", "shared") != expected_attention:
        raise ValueError("v17 arm and attention mode disagree")
    scale = float(dendrites.get("coincidence_scale", 0.1))
    if ((arm == "dendritic" and not scale > 0)
            or (arm == "dendritic_additive" and scale != 0)):
        raise ValueError("v17 dendritic arm and coincidence scale disagree")
    return arm
