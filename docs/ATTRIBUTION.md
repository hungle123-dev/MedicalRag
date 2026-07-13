# Data, models and method attribution

MedicalRAG is original project code. It reimplements published algorithms at the idea level and
does not copy third-party repository code.

- **BioASQ-12B-RAG bundle** — `mattmorgis/bioasq-12b-rag`, pinned source revision
  `6d6add1a6ec2090991386b5ae7608b71fd637bc4`, licensed CC BY-NC-SA 4.0. The local manifest
  additionally pins every raw file by SHA-256. Dataset card:
  <https://huggingface.co/datasets/mattmorgis/bioasq-12b-rag>.
- **MedCPT encoders and cross-encoder** — NCBI public-domain model releases. Every downloaded
  model revision is resolved to an immutable Hugging Face commit in index metadata. Model card:
  <https://huggingface.co/ncbi/MedCPT-Article-Encoder>. Paper: Jin et al., “MedCPT: Contrastive
  Pre-trained Transformers with Large-scale PubMed Search Logs for Zero-shot Biomedical
  Information Retrieval,” *Bioinformatics* (2023), <https://pmc.ncbi.nlm.nih.gov/articles/PMC10627406/>.
- **Reciprocal Rank Fusion** — Cormack, Clarke and Büttcher (SIGIR 2009), implemented with the
  paper's rank-only `k=60` form. <https://doi.org/10.1145/1571941.1572114>.
- **HyDE** — Gao, Ma, Lin and Callan, “Precise Zero-Shot Dense Retrieval without Relevance
  Labels,” ACL 2023. <https://aclanthology.org/2023.acl-long.99/>. HyDE is a registered contrast,
  not assumed to improve biomedical retrieval.

PubMed is provenance for the title/abstract records in the bundle; this project neither crawls
PubMed nor claims PubMed-wide search. Generated LLM outputs remain subject to the upstream
provider terms configured by the local user and are excluded from Git.
