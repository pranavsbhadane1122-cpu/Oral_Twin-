"""Load project configuration from configs/config.yaml."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path=CONFIG_PATH):
    """Read the YAML config file and return it as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
