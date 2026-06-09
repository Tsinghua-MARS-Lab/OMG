from __future__ import annotations

from typing import Any


def _metric_text(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, dict):
        if value.get("mean") is None:
            return "N/A"
        mean = float(value["mean"])
        std = value.get("std")
        if std is None:
            return f"{mean:.4f}"
        return f"{mean:.4f} +/- {float(std):.4f}"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def generation_benchmark_markdown(
    *,
    rows: list[dict[str, Any]],
    metric_directions: dict[str, str],
    metadata: dict[str, Any],
) -> str:
    metric_names = list(metric_directions.keys())
    best: dict[str, int] = {}
    for metric in metric_names:
        candidates = []
        for idx, row in enumerate(rows):
            value = row.get(metric)
            if isinstance(value, dict):
                value = value.get("mean")
            if isinstance(value, (int, float)):
                candidates.append((idx, float(value)))
        if candidates:
            reverse = metric_directions[metric] == "max"
            best[metric] = sorted(candidates, key=lambda item: item[1], reverse=reverse)[0][0]

    lines = ["# Generation Benchmark Summary", ""]
    for key, value in metadata.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "| ckpt | " + " | ".join(metric_names) + " |"])
    lines.append("|---|" + "|".join("---:" for _ in metric_names) + "|")
    for idx, row in enumerate(rows):
        cells = []
        for metric in metric_names:
            text = _metric_text(row.get(metric))
            if best.get(metric) == idx and text != "N/A":
                text = f"**{text}**"
            cells.append(text)
        lines.append(f"| {row['ckpt']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
