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
    summaries: list[tuple[str, dict[str, Any]]] = []
    for path in sorted((ROOT / "reports" / "runs").glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("status", "").startswith("observed"):
                summaries.append((path.stem, value))
        except (json.JSONDecodeError, OSError):
            continue
    candidate_cards: list[str] = []
    pilot_cards: list[str] = []
    for name, summary in summaries:
        config = summary.get("config", {})
        metrics = "".join(
            f"<tr><td>{html.escape(key)}</td><td>{html.escape(value)}</td></tr>"
            for key, value in _flatten(summary.get("metrics", {}))
        )
        card = (
            f"<article><div class='eyebrow'>{html.escape(str(config.get('family', 'RUN')))} · "
            f"{html.escape(str(config.get('population', '')))}</div>"
            f"<h3>{html.escape(str(config.get('arm', name)))}</h3>"
            f"<p><b>Mục đích:</b> {html.escape(str(config.get('purpose', 'unspecified')))}</p>"
            f"<table><tbody>{metrics}</tbody></table></article>"
        )
        target = pilot_cards if config.get("purpose") == "feasibility_only" else candidate_cards
        target.append(card)
    decision_cards = []
    for path in sorted((ROOT / "reports" / "comparisons").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        bootstrap = value["bootstrap"]
        mde_row = (
            f"<tr><td>MDE 80%</td><td>{value['normal_approx_mde_80']:.6f}</td></tr>"
            if "normal_approx_mde_80" in value
            else ""
        )
        decision_cards.append(
            "<article><div class='eyebrow'>PAIRED COMPARISON</div>"
            f"<h3>{html.escape(path.stem)}</h3>"
            "<table><tbody>"
            f"<tr><td>Metric</td><td>{html.escape(value['metric'])}</td></tr>"
            f"<tr><td>Δ right − left</td><td>"
            f"{bootstrap['mean_delta_right_minus_left']:.6f}</td></tr>"
            f"<tr><td>CI95%</td><td>[{bootstrap['ci95_low']:.6f}, "
            f"{bootstrap['ci95_high']:.6f}]</td></tr>"
            f"<tr><td>Permutation p</td><td>{value['paired_permutation_p']:.6g}</td></tr>"
            f"<tr><td>Paired effect</td><td>{value['paired_effect_size']:.6f}</td></tr>"
            f"{mde_row}"
            "</tbody></table></article>"
        )
    for path in sorted((ROOT / "reports" / "gates").glob("*.json")):
        gate = json.loads(path.read_text(encoding="utf-8"))
        checks = "".join(
            f"<tr><td>{html.escape(key)}</td><td>{'PASS' if value else 'FAIL'}</td></tr>"
            for key, value in gate.get("checks", {}).items()
        )
        decision_cards.append(
            f"<article class='gate {'pass' if gate['passed'] else 'fail'}'>"
            f"<div class='eyebrow'>DECISION GATE</div><h3>{html.escape(gate['gate_id'])}</h3>"
            f"<p><b>{'PASS' if gate['passed'] else 'FAIL'}</b></p>"
            f"<table><tbody>{checks}</tbody></table></article>"
        )
    judge_path = ROOT / "reports" / "judge_sanity.json"
    if judge_path.is_file():
        judge = json.loads(judge_path.read_text(encoding="utf-8"))
        decision_cards.append(
            "<article><div class='eyebrow'>JUDGE CALIBRATION</div>"
            "<h3>Multi-LLM sanity check</h3>"
            f"<p><b>{'PASSED' if judge['passed'] else 'FAILED'}</b></p>"
            f"<p>Models: {html.escape(', '.join(judge['models']))}</p>"
            "<p>Chỉ là proxy tự động; không phải human/physician validation.</p></article>"
        )
    sanity_path = ROOT / "reports" / "panel" / "sanity40.json"
    if sanity_path.is_file():
        sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
        decision_cards.append(
            f"<article class='gate {'pass' if sanity['passed'] else 'fail'}'>"
            "<div class='eyebrow'>JUDGE SANITY40</div><h3>40 deterministic controls</h3>"
            f"<p><b>{'PASS' if sanity['passed'] else 'FAIL'}</b> · "
            f"pass rate {sanity['pass_rate']:.1%} · failure {sanity['failure_rate']:.1%}</p>"
            f"<p>{html.escape(', '.join(sanity['models']))}</p></article>"
        )
    for path in sorted((ROOT / "reports" / "incidents").glob("*.json")):
        incident = json.loads(path.read_text(encoding="utf-8"))
        decision_cards.append(
            "<article class='fail'><div class='eyebrow'>RECORDED INCIDENT</div>"
            f"<h3>{html.escape(incident['incident_id'])}</h3>"
            f"<p>{incident['failures']}/{incident['rows']} failures; "
            f"first index {incident['first_failure_index_zero_based']}.</p>"
            f"<p>{html.escape(incident['interpretation'])}</p></article>"
        )
    output = destination or ROOT / "reports" / "FINAL_REPORT.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Kết quả MedicalRAG</title>
<style>
body{{font:15px Inter,system-ui;margin:0;background:#f3f6f4;color:#173238}}
main{{max-width:1240px;margin:auto;padding:48px 24px}} h1{{font-size:44px;margin-bottom:10px}}
h2{{margin:42px 0 16px}} .notice{{padding:18px;border-left:4px solid #117c72;background:white}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}}
article{{background:white;border:1px solid #d9e3df;border-radius:12px;padding:22px}}
table{{width:100%;border-collapse:collapse}}
td{{padding:7px;border-bottom:1px solid #e7eeeb;font-size:13px}}
td:last-child{{text-align:right;font-family:ui-monospace,monospace}} small,.eyebrow{{color:#63777a}}
.eyebrow{{font-size:11px;font-weight:800;letter-spacing:.09em}}
.pass{{border-top:4px solid #11856e}} .fail{{border-top:4px solid #bb3d3d}}
details{{margin-top:18px}} summary{{cursor:pointer;font-weight:700}}
</style></head><body><main>
<small>Tạo tự động lúc {datetime.now(UTC).isoformat()}</small>
<h1>Kết quả thực nghiệm MedicalRAG</h1>
<p class="notice"><b>Phạm vi kết luận:</b> chỉ hiển thị artifact đã đo bằng máy. Corpus BioASQ
49.513 abstracts là positive-only, gold-conditioned closed corpus; kết quả không chứng minh
PubMed-wide retrieval, độ an toàn lâm sàng hay khả năng tổng quát y khoa. Exact-answer metrics bị
tắt vì bundle hiện tại không có official exact labels.</p>
<h2>Quyết định có kiểm định</h2><div class="grid">
{"".join(decision_cards) or "<article>Chưa có paired decision.</article>"}</div>
<h2>Candidate/diagnostic runs</h2><div class="grid">
{"".join(candidate_cards) or "<article>Chưa có candidate run.</article>"}</div>
<details><summary>Pilot/feasibility runs — không dùng để chọn winner</summary>
<div class="grid">{"".join(pilot_cards) or "<article>Không có pilot.</article>"}</div></details>
</main></body></html>""",
        encoding="utf-8",
    )
    return output
