# Bias, overfitting and leakage controls

- Development uses the frozen 300-question BioASQ dev sample. The 340-question `eval` split stays locked until retrieval, graph, prompt, model ID, evidence budget and retry policy are frozen. No retuning follows locked access.
- B3 and G2 use identical question IDs, generator snapshot, temperature 0, top-p 1, 512 output-token limit, prompt, retry policy and 1,800-token/8-item evidence budget. Text is placed first and graph/text are rank-interleaved to reduce graph-first position bias.
- The generator never receives gold answers, gold PMID/snippets, pipeline IDs, experiment metrics or competing answers. It is explicitly told not to prefer graph evidence and not to interpret an association as causality.
- Q1/Q2 runtime logic cannot observe gold misses. Q1 was rejected on dev because harmed-query rate exceeded rescued-query rate. Q2 was not promoted after that gate.
- Correctness/completeness judging receives only question, reference and candidate. Faithfulness judging receives only candidate and cited evidence. Both are blinded to pipeline and retrieval scores.
- PrimeKGQA test is accessed only after graph linking/relation canonicalization freeze. Because the RDF compatibility gate failed, SPARQL execution accuracy is not reported as valid; normalized-pattern results remain a component benchmark, not clinical validation.
- Human review uses 100 IDs sampled from locked BioASQ `eval` before locked outputs. Two qualified reviewers see randomized anonymous A/B answers; AI cannot substitute for them.
- Confirmatory claims require paired statistics and must survive equal-budget extra-text and matched random-path controls. API/quota failures cannot silently change the paired population.
