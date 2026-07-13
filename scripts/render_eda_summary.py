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
corpus = eda["bioasq"]["corpus"]
rows += [("bioasq_corpus", "text_chars_mean", corpus["text_chars"]["mean"]),
         ("bioasq_corpus", "text_chars_p95", corpus["text_chars"]["p95"]),
         ("bioasq_corpus", "empty_text", corpus["empty_text"]),
         ("bioasq_corpus", "missing_title", corpus["missing_title"]),
         ("bioasq_corpus", "missing_doi", corpus["missing_doi"])]
for split in ("dev", "eval"):
    item = eda["bioasq"][split]
    rows += [(f"bioasq_{split}", "question_chars_mean", item["question_chars"]["mean"]),
             (f"bioasq_{split}", "answer_chars_mean", item["answer_chars"]["mean"]),
             (f"bioasq_{split}", "gold_passage_id_coverage", item["gold_passage_id_coverage"]),
             (f"bioasq_{split}", "snippet_chars_mean", item["snippet_chars"]["mean"]),
             (f"bioasq_{split}", "empty_answers", item["empty_answers"])]
degree = eda["primekg"]["degree"]
rows += [("primekg", "degree_median", degree["median"]), ("primekg", "degree_p95", degree["p95"]),
         ("primekg", "degree_max", degree["max"]),
         ("primekg", "duplicate_normalized_names", eda["primekg"]["duplicate_normalized_names"]),
         ("primekg", "missing_edge_endpoints", eda["primekg"]["missing_edge_endpoints"])]
for split in ("train", "val", "test"):
    rows.append((f"primekgqa_{split}", "missing_natural_language_question",
                 eda["primekgqa"][split]["missing_natural_language_question"]))
for pair, count in eda["primekgqa"]["exact_question_overlap"].items():
    rows.append(("primekgqa", f"exact_question_overlap_{pair}", count))
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
