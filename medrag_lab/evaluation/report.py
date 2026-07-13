from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medrag_lab.settings import ROOT


def _flatten(value: dict[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    rows = []
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            rows.extend(_flatten(item, name))
        elif isinstance(item, (int, float, str)):
            rows.append((name, str(round(item, 6) if isinstance(item, float) else item)))
    return rows


def build_report(destination: Path | None = None) -> Path:
    summaries = []
    for path in sorted((ROOT / "reports" / "runs").glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("status", "").startswith("observed"):
                summaries.append((path.stem, value))
        except (json.JSONDecodeError, OSError):
            continue
    cards = []
    for name, summary in summaries:
        metrics = "".join(
            f"<tr><td>{html.escape(key)}</td><td>{html.escape(value)}</td></tr>"
            for key, value in _flatten(summary.get("metrics", {}))
        )
        cards.append(
            f"<article><h3>{html.escape(name)}</h3>"
            f"<p><b>Status:</b> {html.escape(summary['status'])}</p>"
            f"<table><tbody>{metrics}</tbody></table></article>"
        )
    output = destination or ROOT / "reports" / "FINAL_REPORT.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MedicalRAG results</title>
<style>
body{{font:16px system-ui;margin:0;background:#f4f8f6;color:#173238}}
main{{max-width:1150px;margin:auto;padding:48px 24px}} h1{{font-size:42px}}
.notice{{padding:18px;border-left:4px solid #117c72;background:white}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}}
article{{background:white;border:1px solid #d9e3df;border-radius:12px;padding:22px}}
table{{width:100%;border-collapse:collapse}}
td{{padding:7px;border-bottom:1px solid #e7eeeb;font-size:13px}}
td:last-child{{text-align:right;font-family:monospace}} small{{color:#63777a}}
</style></head><body><main>
<small>Generated {datetime.now(UTC).isoformat()}</small>
<h1>MedicalRAG observed results</h1>
<p class="notice">Only machine-observed run artifacts are included. This positive-only BioASQ
candidate pool measures controlled closed-corpus reranking and ideal-answer generation; it does
not establish PubMed-wide retrieval, clinical safety, or medical generalization. Exact-answer
metrics remain disabled because official exact labels are absent.</p>
<div class="grid">
{"".join(cards) or "<article>No completed observed runs found.</article>"}
</div></main></body></html>""",
        encoding="utf-8",
    )
    return output
