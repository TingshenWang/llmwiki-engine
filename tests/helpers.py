from pathlib import Path


def copy_fixture_raw(vault: Path, fixture_raw: Path) -> Path:
    target = vault / "raw" / fixture_raw.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(fixture_raw.read_bytes())
    return target
