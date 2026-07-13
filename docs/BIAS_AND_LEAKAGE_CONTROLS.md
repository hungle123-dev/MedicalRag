# Bias, overfitting and leakage controls

- Development uses the frozen 300-question BioASQ dev sample. EDA and ID freezing may read eval metadata; “locked” means eval labels/output scores are never used to tune and the frozen pipeline executes once. No retuning follows that execution.
- B3/G2/X1/X2 use identical IDs, generator, prompt, output limit and retry policy. Per question, all arms match B3's actual whitespace-word count; X1 matches graph slot count/length with text and X2 matches hop count/path count/nearest length.
- The generator never receives gold answers, gold PMID/snippets, pipeline IDs, experiment metrics or competing answers. It is explicitly told not to prefer graph evidence and not to interpret an association as causality.
- Q1/Q2 runtime logic cannot observe gold misses. Q1 was rejected on dev because harmed-query rate exceeded rescued-query rate. Q2 was not promoted after that gate.
- Correctness/completeness judging receives only question, reference and candidate. Faithfulness judging receives only candidate and cited evidence. Both are blinded to pipeline and retrieval scores.
- PrimeKGQA test is accessed only after graph linking/relation canonicalization freeze. Because the RDF compatibility gate failed, SPARQL execution accuracy is not reported as valid; normalized-pattern results remain a component benchmark, not clinical validation.
- Human review uses 100 IDs sampled from locked BioASQ `eval` before locked outputs. Two qualified reviewers see randomized anonymous A/B answers; AI cannot substitute for them.
- Confirmatory claims require paired statistics and must survive equal-budget extra-text and matched random-path controls. API/quota failures cannot silently change the paired population.
