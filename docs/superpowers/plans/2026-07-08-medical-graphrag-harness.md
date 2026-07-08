# Medical Graph RAG Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable, TDD-covered harness that compares KG-RAG strategies (E0 no-retrieval, E1 text-RAG, E2 KG-triple) for medical QA, on a synthetic MIRAGE-shaped fixture with a mock LLM — so the whole pipeline runs green with no API key / UMLS license / dataset download.

**Architecture:** A small pluggable pipeline. `LLMClient` (protocol) has a `MockLLM` impl for tests. `Retriever` (protocol) has `NullRetriever` (E0), `TextRetriever` (E1, BM25 over a corpus), `TripleRetriever` (E2, entity→triple lookup). An `Arm` composes retriever + prompt builder + LLM into `answer(question) -> Prediction`. A `runner` loops MIRAGE-shaped questions, scores MCQA accuracy, writes a results JSON. Real MIRAGE / UMLS / API adapters slot behind the same protocols later.

**Tech Stack:** Python 3.12, pytest, `rank-bm25` (pure-python BM25, no native deps), stdlib `json`/`dataclasses`. No GPU, no network at test time.

## Global Constraints

- Python 3.12 (available: 3.12.10).
- No network / API key / UMLS license required for any test in this plan. Real adapters are stubbed behind protocols and are OUT OF SCOPE here.
- Every arm consumes the SAME knowledge source in real runs (design doc mục 6). In this MVP the shared source is the fixture corpus; UMLS is only an entity-normalizer, never a knowledge source.
- MCQA scoring only (design doc: MCE). Grounding metric is a LATER task, not in this plan.
- TDD: failing test first, minimal impl, green, commit. One dependency added only when a test needs it.
- Dataclasses for all data-carrying types. Protocols (`typing.Protocol`) for pluggable seams.

---
## File Structure

- `src/medgraphrag/__init__.py` — package marker
- `src/medgraphrag/types.py` — dataclasses: `Question`, `Prediction`, `RetrievedItem`
- `src/medgraphrag/llm.py` — `LLMClient` protocol + `MockLLM`
- `src/medgraphrag/retrievers.py` — `Retriever` protocol + `NullRetriever`, `TextRetriever`, `TripleRetriever`
- `src/medgraphrag/arms.py` — `Arm` composing retriever+prompt+LLM; factory `build_arm(name, ...)`
- `src/medgraphrag/scoring.py` — `accuracy(preds, gold)`
- `src/medgraphrag/runner.py` — `run_arm(arm, questions) -> RunResult`; `save_results`
- `tests/fixtures/mini_mirage.py` — 6-question synthetic MCQA set + tiny corpus + triples
- `tests/test_*.py` — one per module
- `pyproject.toml` — deps (pytest, rank-bm25), pytest config, src layout
- `requirements.txt` — pinned deps

**Data shapes (locked here, used by every task):**

```python
@dataclass(frozen=True)
class Question:
    qid: str
    text: str
    options: dict[str, str]   # {"A": "...", "B": "...", ...}
    answer: str               # gold key, e.g. "B"

@dataclass(frozen=True)
class RetrievedItem:
    content: str
    score: float
    source: str               # "corpus:<docid>" or "triple:<h|r|t>"

@dataclass(frozen=True)
class Prediction:
    qid: str
    choice: str               # predicted option key, e.g. "B"
    evidence: tuple[RetrievedItem, ...]  # what retriever supplied (grounding later)
```

---

### Task 1: Project scaffold + package + data types

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `src/medgraphrag/__init__.py`
- Create: `src/medgraphrag/types.py`
- Test: `tests/test_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Question`, `RetrievedItem`, `Prediction` dataclasses (shapes above); importable as `from medgraphrag.types import Question, RetrievedItem, Prediction`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "medgraphrag"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["rank-bm25==0.2.2"]

[project.optional-dependencies]
dev = ["pytest==8.3.4"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Write `requirements.txt`**

```
rank-bm25==0.2.2
pytest==8.3.4
```

- [ ] **Step 3: Create empty `src/medgraphrag/__init__.py`**

```python
```

- [ ] **Step 4: Write the failing test**

```python
# tests/test_types.py
from medgraphrag.types import Question, RetrievedItem, Prediction


def test_question_holds_mcqa_fields():
    q = Question(qid="q1", text="What?", options={"A": "x", "B": "y"}, answer="B")
    assert q.answer == "B"
    assert q.options["A"] == "x"


def test_prediction_carries_evidence():
    item = RetrievedItem(content="fact", score=1.0, source="corpus:d1")
    p = Prediction(qid="q1", choice="B", evidence=(item,))
    assert p.choice == "B"
    assert p.evidence[0].source == "corpus:d1"
```

- [ ] **Step 5: Run test to verify it fails**

Run: `python -m pytest tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'medgraphrag.types'`

- [ ] **Step 6: Write `src/medgraphrag/types.py`**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    qid: str
    text: str
    options: dict[str, str]
    answer: str


@dataclass(frozen=True)
class RetrievedItem:
    content: str
    score: float
    source: str


@dataclass(frozen=True)
class Prediction:
    qid: str
    choice: str
    evidence: tuple[RetrievedItem, ...]
```

- [ ] **Step 7: Install dev deps + run test to verify it passes**

Run: `python -m pip install -e ".[dev]" && python -m pytest tests/test_types.py -v`
Expected: PASS (2 passed)

- [ ] **Step 8: Commit**

```bash
git init
git add pyproject.toml requirements.txt src/medgraphrag/__init__.py src/medgraphrag/types.py tests/test_types.py
git commit -m "feat: project scaffold + core data types"
```

### Task 2: MockLLM + LLMClient protocol

**Files:**
- Create: `src/medgraphrag/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `LLMClient` protocol with `choose(question_text: str, options: dict[str,str], context: str) -> str` (returns an option key). `MockLLM(rules: dict[str,str] = None, default: str = "A")` — deterministic: if any option-key's option-text appears verbatim in `context`, return that key; else if `qid`-free `rules` maps a substring of question_text, use it; else `default`. This lets tests assert "retrieval changed the answer".

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm.py
from medgraphrag.llm import MockLLM


def test_mock_picks_option_whose_text_is_in_context():
    llm = MockLLM(default="A")
    opts = {"A": "aspirin", "B": "penicillin"}
    choice = llm.choose("treatment?", opts, context="give penicillin now")
    assert choice == "B"


def test_mock_falls_back_to_default_without_context_hit():
    llm = MockLLM(default="A")
    opts = {"A": "aspirin", "B": "penicillin"}
    choice = llm.choose("treatment?", opts, context="no useful text")
    assert choice == "A"


def test_mock_rules_override_default():
    llm = MockLLM(rules={"fever": "B"}, default="A")
    opts = {"A": "aspirin", "B": "penicillin"}
    choice = llm.choose("patient has fever", opts, context="")
    assert choice == "B"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'medgraphrag.llm'`

- [ ] **Step 3: Write `src/medgraphrag/llm.py`**

```python
from typing import Protocol


class LLMClient(Protocol):
    def choose(self, question_text: str, options: dict[str, str], context: str) -> str:
        ...


class MockLLM:
    def __init__(self, rules: dict[str, str] | None = None, default: str = "A"):
        self._rules = rules or {}
        self._default = default

    def choose(self, question_text: str, options: dict[str, str], context: str) -> str:
        ctx = context.lower()
        for key, text in options.items():
            if text.lower() in ctx:
                return key
        for needle, key in self._rules.items():
            if needle.lower() in question_text.lower():
                return key
        return self._default
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_llm.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/medgraphrag/llm.py tests/test_llm.py
git commit -m "feat: MockLLM + LLMClient protocol"
```

---

### Task 3: Synthetic mini-MIRAGE fixture

**Files:**
- Create: `tests/fixtures/__init__.py` (empty)
- Create: `tests/fixtures/mini_mirage.py`
- Test: `tests/test_fixture.py`

**Interfaces:**
- Consumes: `Question` (Task 1).
- Produces: `QUESTIONS: list[Question]` (6 MCQA items), `CORPUS: dict[str, str]` (docid→text; each answer's option-text appears in exactly one doc), `TRIPLES: list[tuple[str,str,str]]` (head, relation, tail; tails contain answer option-text). Mirrors MIRAGE MCE shape; NOT real MIRAGE data.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fixture.py
from tests.fixtures.mini_mirage import QUESTIONS, CORPUS, TRIPLES
from medgraphrag.types import Question


def test_six_questions_all_valid_mcqa():
    assert len(QUESTIONS) == 6
    for q in QUESTIONS:
        assert isinstance(q, Question)
        assert q.answer in q.options


def test_each_answer_supported_somewhere_in_corpus():
    joined = " ".join(CORPUS.values()).lower()
    for q in QUESTIONS:
        assert q.options[q.answer].lower() in joined


def test_triples_are_triples():
    for t in TRIPLES:
        assert len(t) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fixture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.fixtures.mini_mirage'`

- [ ] **Step 3: Write `tests/fixtures/__init__.py` (empty) and `tests/fixtures/mini_mirage.py`**

```python
# tests/fixtures/mini_mirage.py
from medgraphrag.types import Question

QUESTIONS = [
    Question("q1", "First-line antibiotic for strep throat?",
             {"A": "penicillin", "B": "insulin", "C": "warfarin", "D": "aspirin"}, "A"),
    Question("q2", "Hormone that lowers blood glucose?",
             {"A": "glucagon", "B": "insulin", "C": "cortisol", "D": "thyroxine"}, "B"),
    Question("q3", "Anticoagulant that inhibits vitamin K?",
             {"A": "aspirin", "B": "heparin", "C": "warfarin", "D": "penicillin"}, "C"),
    Question("q4", "Drug that irreversibly inhibits COX to reduce platelet aggregation?",
             {"A": "aspirin", "B": "insulin", "C": "glucagon", "D": "heparin"}, "A"),
    Question("q5", "Hormone that raises blood glucose?",
             {"A": "insulin", "B": "glucagon", "C": "warfarin", "D": "penicillin"}, "B"),
    Question("q6", "Parenteral anticoagulant potentiating antithrombin III?",
             {"A": "warfarin", "B": "aspirin", "C": "heparin", "D": "insulin"}, "C"),
]

CORPUS = {
    "d1": "Penicillin is the first-line antibiotic for streptococcal pharyngitis.",
    "d2": "Insulin is the hormone that lowers blood glucose after meals.",
    "d3": "Warfarin is an oral anticoagulant that inhibits vitamin K epoxide reductase.",
    "d4": "Aspirin irreversibly inhibits COX and reduces platelet aggregation.",
    "d5": "Glucagon is the hormone that raises blood glucose during fasting.",
    "d6": "Heparin is a parenteral anticoagulant that potentiates antithrombin III.",
}

TRIPLES = [
    ("strep throat", "first_line_treatment", "penicillin"),
    ("blood glucose", "lowered_by", "insulin"),
    ("vitamin K", "inhibited_by", "warfarin"),
    ("platelet aggregation", "reduced_by", "aspirin"),
    ("blood glucose", "raised_by", "glucagon"),
    ("antithrombin III", "potentiated_by", "heparin"),
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fixture.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/__init__.py tests/fixtures/mini_mirage.py tests/test_fixture.py
git commit -m "test: synthetic mini-MIRAGE fixture"
```

### Task 4: Retrievers (Null, Text/BM25, Triple)

**Files:**
- Create: `src/medgraphrag/retrievers.py`
- Test: `tests/test_retrievers.py`

**Interfaces:**
- Consumes: `RetrievedItem` (Task 1); fixture `CORPUS`, `TRIPLES` (Task 3).
- Produces: `Retriever` protocol with `retrieve(query: str, k: int) -> list[RetrievedItem]`.
  - `NullRetriever()` → always `[]` (E0).
  - `TextRetriever(corpus: dict[str,str])` → BM25 over corpus docs, top-k, `source="corpus:<docid>"`.
  - `TripleRetriever(triples: list[tuple[str,str,str]])` → token-overlap match of query against `head`+`tail`; returns items with `content="<h> <r> <t>"`, `source="triple:<h>|<r>|<t>"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retrievers.py
from medgraphrag.retrievers import NullRetriever, TextRetriever, TripleRetriever
from tests.fixtures.mini_mirage import CORPUS, TRIPLES


def test_null_returns_nothing():
    assert NullRetriever().retrieve("anything", k=5) == []


def test_text_retriever_ranks_relevant_doc_first():
    r = TextRetriever(CORPUS)
    items = r.retrieve("antibiotic for strep throat", k=1)
    assert len(items) == 1
    assert "penicillin" in items[0].content.lower()
    assert items[0].source.startswith("corpus:")


def test_triple_retriever_matches_on_overlap():
    r = TripleRetriever(TRIPLES)
    items = r.retrieve("what lowers blood glucose", k=1)
    assert len(items) == 1
    assert "insulin" in items[0].content.lower()
    assert items[0].source.startswith("triple:")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retrievers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'medgraphrag.retrievers'`

- [ ] **Step 3: Write `src/medgraphrag/retrievers.py`**

```python
import re
from typing import Protocol

from rank_bm25 import BM25Okapi

from medgraphrag.types import RetrievedItem


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[RetrievedItem]:
        ...


class NullRetriever:
    def retrieve(self, query: str, k: int) -> list[RetrievedItem]:
        return []


class TextRetriever:
    def __init__(self, corpus: dict[str, str]):
        self._ids = list(corpus.keys())
        self._docs = [corpus[i] for i in self._ids]
        self._bm25 = BM25Okapi([_tokenize(d) for d in self._docs])

    def retrieve(self, query: str, k: int) -> list[RetrievedItem]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked[:k]:
            out.append(RetrievedItem(content=self._docs[i], score=float(scores[i]),
                                     source=f"corpus:{self._ids[i]}"))
        return out


class TripleRetriever:
    def __init__(self, triples: list[tuple[str, str, str]]):
        self._triples = triples

    def retrieve(self, query: str, k: int) -> list[RetrievedItem]:
        q = set(_tokenize(query))
        scored = []
        for h, r, t in self._triples:
            terms = set(_tokenize(h)) | set(_tokenize(t))
            overlap = len(q & terms)
            if overlap:
                scored.append((overlap, h, r, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for overlap, h, r, t in scored[:k]:
            out.append(RetrievedItem(content=f"{h} {r} {t}", score=float(overlap),
                                     source=f"triple:{h}|{r}|{t}"))
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_retrievers.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/medgraphrag/retrievers.py tests/test_retrievers.py
git commit -m "feat: Null/Text(BM25)/Triple retrievers"
```

---

### Task 5: Arm (retriever + prompt + LLM) and factory

**Files:**
- Create: `src/medgraphrag/arms.py`
- Test: `tests/test_arms.py`

**Interfaces:**
- Consumes: `Question`, `Prediction`, `RetrievedItem` (Task 1); `LLMClient` (Task 2); `Retriever`, `NullRetriever`, `TextRetriever`, `TripleRetriever` (Task 4); fixture `CORPUS`, `TRIPLES` (Task 3).
- Produces: `Arm(retriever: Retriever, llm: LLMClient, k: int = 3)` with `answer(q: Question) -> Prediction` — retrieves, joins item contents into `context`, calls `llm.choose`, returns `Prediction` with evidence. `build_arm(name: str, corpus, triples, llm, k=3) -> Arm` where `name in {"E0","E1","E2"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_arms.py
from medgraphrag.arms import Arm, build_arm
from medgraphrag.llm import MockLLM
from tests.fixtures.mini_mirage import QUESTIONS, CORPUS, TRIPLES


def test_e0_uses_no_context_falls_to_default():
    # MockLLM default "A"; q2 gold is "B" -> without retrieval it should miss
    arm = build_arm("E0", CORPUS, TRIPLES, MockLLM(default="A"))
    pred = arm.answer(QUESTIONS[1])  # q2
    assert pred.choice == "A"
    assert pred.evidence == ()


def test_e1_text_retrieval_supplies_context_and_fixes_answer():
    arm = build_arm("E1", CORPUS, TRIPLES, MockLLM(default="A"))
    pred = arm.answer(QUESTIONS[1])  # q2 gold "B" (insulin)
    assert pred.choice == "B"
    assert any(e.source.startswith("corpus:") for e in pred.evidence)


def test_e2_triple_retrieval_supplies_context():
    arm = build_arm("E2", CORPUS, TRIPLES, MockLLM(default="A"))
    pred = arm.answer(QUESTIONS[1])  # q2 gold "B"
    assert pred.choice == "B"
    assert any(e.source.startswith("triple:") for e in pred.evidence)


def test_build_arm_rejects_unknown_name():
    import pytest
    with pytest.raises(ValueError):
        build_arm("E9", CORPUS, TRIPLES, MockLLM())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_arms.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'medgraphrag.arms'`

- [ ] **Step 3: Write `src/medgraphrag/arms.py`**

```python
from medgraphrag.types import Question, Prediction
from medgraphrag.llm import LLMClient
from medgraphrag.retrievers import (
    Retriever, NullRetriever, TextRetriever, TripleRetriever,
)


class Arm:
    def __init__(self, retriever: Retriever, llm: LLMClient, k: int = 3):
        self._retriever = retriever
        self._llm = llm
        self._k = k

    def answer(self, q: Question) -> Prediction:
        items = self._retriever.retrieve(q.text, self._k)
        context = "\n".join(i.content for i in items)
        choice = self._llm.choose(q.text, q.options, context)
        return Prediction(qid=q.qid, choice=choice, evidence=tuple(items))


def build_arm(name, corpus, triples, llm, k: int = 3) -> Arm:
    if name == "E0":
        return Arm(NullRetriever(), llm, k)
    if name == "E1":
        return Arm(TextRetriever(corpus), llm, k)
    if name == "E2":
        return Arm(TripleRetriever(triples), llm, k)
    raise ValueError(f"unknown arm: {name}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_arms.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/medgraphrag/arms.py tests/test_arms.py
git commit -m "feat: Arm composition + build_arm factory (E0/E1/E2)"
```

### Task 6: Scoring (MCQA accuracy)

**Files:**
- Create: `src/medgraphrag/scoring.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: `Question`, `Prediction` (Task 1).
- Produces: `accuracy(preds: list[Prediction], questions: list[Question]) -> float` — matches by `qid`, fraction where `pred.choice == question.answer`. Raises `ValueError` if a prediction has no matching question or counts mismatch.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scoring.py
import pytest
from medgraphrag.scoring import accuracy
from medgraphrag.types import Question, Prediction


def _q(qid, ans):
    return Question(qid, "?", {"A": "a", "B": "b"}, ans)


def _p(qid, choice):
    return Prediction(qid, choice, ())


def test_all_correct_is_one():
    qs = [_q("q1", "A"), _q("q2", "B")]
    ps = [_p("q1", "A"), _p("q2", "B")]
    assert accuracy(ps, qs) == 1.0


def test_half_correct():
    qs = [_q("q1", "A"), _q("q2", "B")]
    ps = [_p("q1", "A"), _p("q2", "A")]
    assert accuracy(ps, qs) == 0.5


def test_mismatched_qid_raises():
    qs = [_q("q1", "A")]
    ps = [_p("qX", "A")]
    with pytest.raises(ValueError):
        accuracy(ps, qs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'medgraphrag.scoring'`

- [ ] **Step 3: Write `src/medgraphrag/scoring.py`**

```python
from medgraphrag.types import Question, Prediction


def accuracy(preds: list[Prediction], questions: list[Question]) -> float:
    gold = {q.qid: q.answer for q in questions}
    if len(preds) != len(questions):
        raise ValueError("prediction/question count mismatch")
    correct = 0
    for p in preds:
        if p.qid not in gold:
            raise ValueError(f"no gold for qid {p.qid}")
        if p.choice == gold[p.qid]:
            correct += 1
    return correct / len(questions)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scoring.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/medgraphrag/scoring.py tests/test_scoring.py
git commit -m "feat: MCQA accuracy scoring"
```

---

### Task 7: Runner — run an arm, score, save results

**Files:**
- Create: `src/medgraphrag/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `Question`, `Prediction` (Task 1); `Arm` (Task 5); `accuracy` (Task 6).
- Produces:
  - `@dataclass RunResult`: `arm_name: str`, `accuracy: float`, `predictions: list[Prediction]`.
  - `run_arm(arm_name: str, arm, questions: list[Question]) -> RunResult`.
  - `save_results(results: list[RunResult], path: str) -> None` — writes JSON `[{"arm": name, "accuracy": x, "n": count}, ...]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner.py
import json
from medgraphrag.runner import run_arm, save_results, RunResult
from medgraphrag.arms import build_arm
from medgraphrag.llm import MockLLM
from tests.fixtures.mini_mirage import QUESTIONS, CORPUS, TRIPLES


def test_run_arm_returns_result_with_accuracy():
    arm = build_arm("E1", CORPUS, TRIPLES, MockLLM(default="A"))
    res = run_arm("E1", arm, QUESTIONS)
    assert isinstance(res, RunResult)
    assert res.arm_name == "E1"
    assert 0.0 <= res.accuracy <= 1.0
    assert len(res.predictions) == len(QUESTIONS)


def test_e1_beats_e0_on_fixture():
    e0 = run_arm("E0", build_arm("E0", CORPUS, TRIPLES, MockLLM(default="A")), QUESTIONS)
    e1 = run_arm("E1", build_arm("E1", CORPUS, TRIPLES, MockLLM(default="A")), QUESTIONS)
    assert e1.accuracy > e0.accuracy


def test_save_results_writes_json(tmp_path):
    res = run_arm("E1", build_arm("E1", CORPUS, TRIPLES, MockLLM()), QUESTIONS)
    out = tmp_path / "results.json"
    save_results([res], str(out))
    data = json.loads(out.read_text())
    assert data[0]["arm"] == "E1"
    assert data[0]["n"] == len(QUESTIONS)
    assert "accuracy" in data[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'medgraphrag.runner'`

- [ ] **Step 3: Write `src/medgraphrag/runner.py`**

```python
import json
from dataclasses import dataclass

from medgraphrag.types import Question, Prediction
from medgraphrag.scoring import accuracy


@dataclass
class RunResult:
    arm_name: str
    accuracy: float
    predictions: list[Prediction]


def run_arm(arm_name: str, arm, questions: list[Question]) -> RunResult:
    preds = [arm.answer(q) for q in questions]
    acc = accuracy(preds, questions)
    return RunResult(arm_name=arm_name, accuracy=acc, predictions=preds)


def save_results(results: list[RunResult], path: str) -> None:
    payload = [
        {"arm": r.arm_name, "accuracy": r.accuracy, "n": len(r.predictions)}
        for r in results
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_runner.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/medgraphrag/runner.py tests/test_runner.py
git commit -m "feat: runner + results serialization"
```

### Task 8: CLI entrypoint + full-suite green

**Files:**
- Create: `src/medgraphrag/__main__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `build_arm` (Task 5), `run_arm`/`save_results` (Task 7), fixture (Task 3).
- Produces: `main(argv: list[str] | None = None) -> int` — runs arms E0,E1,E2 on the fixture, prints one `NAME\taccuracy` line each, writes `results.json` to cwd, returns 0. Runnable as `python -m medgraphrag`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
from medgraphrag.__main__ import main


def test_main_runs_all_arms_and_returns_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    for name in ("E0", "E1", "E2"):
        assert name in out
    assert (tmp_path / "results.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'medgraphrag.__main__'`

- [ ] **Step 3: Write `src/medgraphrag/__main__.py`**

```python
import sys

from medgraphrag.arms import build_arm
from medgraphrag.llm import MockLLM
from medgraphrag.runner import run_arm, save_results
from tests.fixtures.mini_mirage import QUESTIONS, CORPUS, TRIPLES


def main(argv: list[str] | None = None) -> int:
    llm = MockLLM(default="A")
    results = []
    for name in ("E0", "E1", "E2"):
        arm = build_arm(name, CORPUS, TRIPLES, llm)
        res = run_arm(name, arm, QUESTIONS)
        results.append(res)
        print(f"{name}\t{res.accuracy:.3f}")
    save_results(results, "results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

Note: importing the fixture from `src` code is deliberate for the MVP demo. Task 9 (future, out of this plan) replaces it with a real MIRAGE loader behind a `load_questions()` seam.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the FULL suite + the CLI for real**

Run: `python -m pytest -v && python -m medgraphrag`
Expected: all tests PASS; CLI prints E0/E1/E2 accuracies (E1 and E2 > E0) and writes `results.json`.

- [ ] **Step 6: Commit**

```bash
git add src/medgraphrag/__main__.py tests/test_cli.py
git commit -m "feat: CLI entrypoint running E0/E1/E2 on fixture"
```

---

## Out of Scope (future plans, behind existing seams)

- **Real MIRAGE loader** — `load_questions()` returning `list[Question]` from the MIRAGE HF dataset; swaps the fixture import in `__main__`.
- **Real LLM adapter** — `LLMClient` impl calling an API (needs key).
- **E3 (MedGraphRAG-inspired) + E4 (HippoRAG)** — new `Retriever` impls; scispaCy+UMLS entity normalizer.
- **Grounding metric** — rubric 0/1/2 evaluator + κ audit (design doc mục 7).
- **Ablations** — CoT×KG 2×2, graph depth, single/multi-hop split.

## Self-Review

- **Spec coverage:** MVP scope (framework + E0/E1/E2 on synthetic MIRAGE-shaped fixture + mock LLM, MCQA accuracy, no key/license) fully covered by Tasks 1–8. E3/E4/grounding/ablation explicitly deferred behind protocols — matches the "MVP harness runnable now" decision.
- **Placeholder scan:** every code/test step has complete code; no TBD/TODO.
- **Type consistency:** `Question`/`Prediction`/`RetrievedItem` shapes fixed in Task 1 and used verbatim through Task 8; `retrieve(query,k)`, `choose(question_text,options,context)`, `answer(q)`, `build_arm(name,corpus,triples,llm,k)`, `run_arm(arm_name,arm,questions)`, `accuracy(preds,questions)` names consistent across tasks.





