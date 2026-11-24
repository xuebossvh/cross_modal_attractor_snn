"""Model arms shared by the v17 planner, training and exporter."""

ARMS = {
    "shared_sum": (False, False),
    "three_state": (True, False),
    "mult": (False, True),
    "main": (True, True),
}


def configure_arm(cfg, arm):
    private, attention = ARMS[arm]
    cfg["v17"]["arm"] = arm
    cfg["v17"]["private_index"]["enabled"] = private
    cfg["v17"]["mult"]["enabled"] = attention


def verify_arm(cfg):
    settings = cfg["v17"]
    arm = settings.get("arm")
    active = (settings["private_index"]["enabled"], settings["mult"]["enabled"])
    if ARMS.get(arm) != active:
        raise ValueError("v17 arm and module switches disagree")
    return arm
