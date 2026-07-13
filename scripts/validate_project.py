"""Fail fast on malformed source-of-truth documents and stale placeholders."""
from html.parser import HTMLParser
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]
html_files = list((ROOT / "docs").glob("*.html"))
for path in html_files:
    text = path.read_text(encoding="utf-8")
    parser = HTMLParser(); parser.feed(text); parser.close()
    ids = re.findall(r'\bid="([^"]+)"', text)
    assert len(ids) == len(set(ids)), f"duplicate HTML id in {path}"
yaml_files = list((ROOT / "configs").glob("*.yaml"))
for path in yaml_files: yaml.safe_load(path.read_text(encoding="utf-8"))
protocol = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text(encoding="utf-8"))
pipelines = yaml.safe_load((ROOT / "configs/pipelines.yaml").read_text(encoding="utf-8"))
assert set(pipelines["pipelines"]) == {"B0", "B1", "B2", "B3", "G1", "G2"}
assert protocol["locked_tests"]["bioasq"]["pipelines"] == ["B3", "G2"]
for path in [ROOT / "README.md", ROOT / "configs/protocol.yaml", *html_files]:
    text = path.read_text(encoding="utf-8")
    assert "MUST_VERIFY" not in text and "TBD_COMMIT" not in text, f"stale placeholder in {path}"
print(f"validated {len(html_files)} HTML files and {len(yaml_files)} YAML contracts")
