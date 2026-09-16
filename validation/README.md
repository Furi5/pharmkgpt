# Validation

Offline verification of completed analyses: required inputs, gold labels, per-question predictions and retrieval rankings, anonymous ratings, statistical code, and figures.
All verification inputs are included. Graph construction and model inference are outside the scope of this package.

```bash
python3 validation/verify.py
```

| Directory | Contents |
| --- | --- |
| data | Shared questions, options, gold labels, and source PMIDs |
| pubmedqa | PubMedQA external validation |
| kg_ablation | KG component ablation |
| effect_size | Accuracy and paired effect sizes |
| free_form_qa | Blinded free-form QA evaluation |
| corpus_retrieval | Corpus size comparison |
| gene_case_study | Candidate-gene literature case study |
| itext2kg_kg2rag | iText2KG + KG2RAG comparison |
| external_kb | External knowledge base validation |
| graphrag | Microsoft GraphRAG comparison |
| closed_models | Closed-book model comparison |
| pubtator_provenance | Entity provenance statistics |
| pubtator_ablation | Entity ablation |
| plot | Plotting code, data, and figures |

See each directory's README for analysis commands and `requirements.txt` for Python dependencies. Analysis files document statistical methods, how failures enter denominators, and interpretation limits.
