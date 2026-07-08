import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "arm" not in cfg or "dataset" not in cfg:
        raise ValueError("config must have 'arm' and 'dataset'")
    return cfg
