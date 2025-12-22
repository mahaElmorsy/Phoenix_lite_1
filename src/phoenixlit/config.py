"""phoenixlit.config

load config.yaml once and pass a Config object around.

"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:

    raw: dict

    def get(self, key, default=None):
        """Return a nested value using dot-notation.
        If the key does not exist, we return `default`.
        """
        cur = self.raw
        for part in str(key).split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur


def load_config(path):
    """Load YAML file and return Config."""
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml root must be a mapping (dict).")
    return Config(raw=data)
