# Human medical review protocol

Population: 100 simple-random IDs from the locked BioASQ `eval` split, frozen in `data/manifests/bioasq_human_100.json` before any locked output existed. Do not replace questions after viewing outputs.

Two medically qualified reviewers independently receive question, reference answer, anonymized answer A/B, and each answer's cited evidence bundle. Literature snippets and structured graph paths remain visible so usefulness can be judged, but pipeline ID, retrieval score and competing-system metrics stay hidden. A/B order is randomized independently per reviewer. Each reviewer records:

- correctness: 0 (wrong/major unsupported conclusion), 1 (core conclusion right with minor omission/error), 2 (all medically important points right);
- completeness: 0 (most essential points missing), 1 (one or more essential points missing), 2 (essential points covered);
- graph usefulness: `supports`, `partial`, `irrelevant`, or `misleading`;
- graph effect: `fixed`, `unchanged`, or `harmed`;
- free-text rationale and primary error code.

Keep raw labels, calculate weighted Cohen's kappa and a confusion matrix, then adjudicate disagreements without overwriting raw values. LLM draft labels may assist triage but never count as either human reviewer. Full-set judge scores are confirmatory only if weighted kappa is at least 0.60.

External blocker: this repository cannot impersonate a physician. Human sign-off remains incomplete until two qualified reviewers submit labels.
