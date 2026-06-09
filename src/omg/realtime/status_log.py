from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def append_jsonl(path: str | Path | None, event: dict[str, Any]) -> None:
    if path is None:
        return
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"wall_time": time.time(), **event}
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
