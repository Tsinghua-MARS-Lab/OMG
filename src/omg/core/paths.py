from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root() / path
