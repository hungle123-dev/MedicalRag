# Human medical review protocol

Population: 100 simple-random IDs from the locked BioASQ `eval` split, frozen in `data/manifests/bioasq_human_100.json` before any locked output existed. Do not replace questions after viewing outputs.

Two medically qualified reviewers independently receive question, reference answer, anonymized answer A/B, and each answer's cited evidence bundle. Literature snippets and structured graph paths remain visible so usefulness can be judged, but pipeline ID, retrieval score and competing-system metrics stay hidden. A/B order is randomized independently per reviewer. Each reviewer records:

- correctness: 0 (wrong/major unsupported conclusion), 1 (core conclusion right with minor omission/error), 2 (all medically important points right);
- completeness: 0 (most essential points missing), 1 (one or more essential points missing), 2 (essential points covered);
- graph usefulness: `supports`, `partial`, `irrelevant`, `misleading`, or `not_applicable`;
- medical harm: `none`, `minor`, or `major` for each answer;
- pair preference: answer A, tie, or answer B;
- free-text per-answer and pair rationale plus primary error code.

Keep raw labels; calculate weighted Cohen's kappa/confusion matrices separately for correctness, completeness and harm, then adjudicate disagreements without overwriting raw values. The analyzer reports paired human G2−B3 correctness and harm-rate bootstrap CIs, graph-usefulness/error counts, and emits an adjudication CSV. LLM draft labels may assist triage but never count as either reviewer. Judge scale-up requires judge–human weighted kappa ≥0.60 against adjudicated labels on the same sample; human–human agreement alone is not judge validity.

External blocker: this repository cannot impersonate a physician. Human sign-off remains incomplete until two qualified reviewers submit labels.
