"""Render a dependency-free EDA CSV and SVG from the canonical JSON report."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
eda = json.loads((ROOT / "data/manifests/eda.json").read_text(encoding="utf-8"))
rows = [("bioasq", "corpus", eda["bioasq"]["corpus"]["rows"]),
        ("bioasq", "dev", eda["bioasq"]["dev"]["rows"]),
        ("bioasq", "eval", eda["bioasq"]["eval"]["rows"]),
        ("primekg", "nodes", eda["primekg"]["nodes"]), ("primekg", "edges", eda["primekg"]["edges"])]
rows += [("primekgqa", split, eda["primekgqa"][split]["rows"]) for split in ("train", "val", "test")]
with (ROOT / "data/manifests/eda_summary.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream); writer.writerow(("dataset", "metric", "value")); writer.writerows(rows)

types = ["factoid", "yesno", "list", "summary"]
dev, test = eda["bioasq"]["dev"]["question_types"], eda["bioasq"]["eval"]["question_types"]
maximum = max(dev.values()); bars = []
for index, label in enumerate(types):
    y = 45 + index * 60
    bars.append(f'<text x="10" y="{y+16}" font-size="14">{label}</text>')
    for offset, values, color, name in ((0, dev, "#08765d", "dev"), (22, test, "#e6a426", "eval")):
        width = 600 * values[label] / maximum
        bars.append(f'<rect x="110" y="{y+offset}" width="{width:.1f}" height="18" fill="{color}" rx="3"/>')
        bars.append(f'<text x="{120+width:.1f}" y="{y+offset+14}" font-size="12">{name} {values[label]}</text>')
svg = '<svg xmlns="http://www.w3.org/2000/svg" width="820" height="300" role="img" aria-label="BioASQ question type counts"><rect width="100%" height="100%" fill="white"/><text x="10" y="24" font-size="18" font-weight="bold">BioASQ question types</text>' + "".join(bars) + "</svg>"
assets = ROOT / "docs/assets"; assets.mkdir(parents=True, exist_ok=True)
(assets / "bioasq_question_types.svg").write_text(svg, encoding="utf-8")
print("wrote EDA CSV and SVG")
