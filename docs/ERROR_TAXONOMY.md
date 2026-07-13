# Error taxonomy

The primary error is the earliest causal failure which, if fixed alone, would plausibly change an incorrect final answer to correct. Later failures are contributing errors. Reviewers use the codes below; “include” and “exclude” prevent double counting.

| Code | Definition and inclusion rule | Exclude when | Example |
|---|---|---|---|
| DATA_SCHEMA | Broken, missing, incompatible or leaked source record | A valid record was processed incorrectly later | PrimeKGQA RDF IRI cannot map to the pinned CSV node ID |
| CHUNK_TRUNC | Relevant content was split or truncated before scoring | Entire relevant document was indexed intact | Answer appears after MedCPT's 512-token cutoff |
| DOC_MISS | No relevant PMID is in the retrieved top-k | A relevant PMID is present but misused | All gold PMIDs rank below 10 |
| QUERY_NOISE | Expansion lowers relevant-document rank | Original query already failed identically | Added entity type dominates BM25 terms |
| ENTITY_LINK | Mention is missed or mapped to the wrong PrimeKG node | Correct seed exists but traversal fails | “cold” links to exposure rather than phenotype |
| GRAPH_COVERAGE | Required entity/relation is absent from pinned PrimeKG | It exists but retriever misses it | No edge represents the needed clinical association |
| REL_DIRECTION | Wrong relation or stored direction is used | Correct edge never entered candidate pool | Contraindication interpreted in reverse |
| HUB_NOISE | High-degree generic nodes dominate paths | Specific but irrelevant path is selected | Anatomy hub displaces the disease node |
| SPURIOUS_PATH | Valid edges form irrelevant or misleading reasoning | Path is relevant but generator misreads it | Two-hop co-neighbor implies unsupported treatment |
| TEXT_GRAPH_DISPLACE | Graph items remove more useful text under matched budget | Total budget was not actually matched | Five short paths leave only three PubMed items |
| ANSWER_REASON | Evidence is adequate but synthesis is medically wrong | Required evidence is absent | Model reverses an association |
| UNSUPPORTED | An atomic claim has no supporting cited evidence | Citation supports the claim but is formatted wrong | New dosage recommendation appears without evidence |
| CITATION_MISMATCH | Citation exists but does not support its attached claim | Citation ID was invented | PMID discusses another disease |
| ABSTENTION_FAIL | System answers confidently despite insufficient/conflicting evidence | Evidence supports the conclusion | G1 answers after no entity links |
| API_RUNTIME | Timeout, quota, cache, transport or config failure | Model returned a substantive bad answer | Gemini quota interrupts one member of a pair |

For each reviewed case store `primary_error`, zero or more `contributing_errors`, inclusion rationale, evidence IDs and reviewer ID. PrimeKG edges are associations unless their relation/source explicitly establishes something stronger; a path alone never proves causality.
