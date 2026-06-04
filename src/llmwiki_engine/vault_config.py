from __future__ import annotations

from pathlib import Path

from .io import read_json, write_json
from .models import VaultConfig


def vault_config_path(vault: Path) -> Path:
    return vault / ".llmwiki" / "config.json"


def read_vault_config(vault: Path) -> VaultConfig:
    path = vault_config_path(vault)
    if not path.exists():
        raise RuntimeError(".llmwiki/config.json is missing; rerun init for the current MVP vault contract.")
    return VaultConfig.model_validate(read_json(path))


def write_default_vault_config(vault: Path) -> VaultConfig:
    config = VaultConfig()
    write_json(vault_config_path(vault), config)
    return config
