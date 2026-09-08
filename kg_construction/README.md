# KG construction

Construct document KGs from PubTator records using the project's customized
iText2KG pipeline: entity extraction, PubTator supplementation, relation
extraction, identifier merging and graph export.

## Run

Python 3.11 and a running Ollama service are required for model inference.

```sh
python -m pip install -r kg_construction/requirements.txt
ollama pull gemma3:27b
ollama pull nomic-embed-text:latest

python kg_construction/build.py \
  --input kg_construction/examples/22815080.txt \
  --output-dir /tmp/pharmkgpt-kg
```

`--input` accepts a PubTator file or directory. Files need matching
`PMID|t|title` and `PMID|a|abstract` headers; entity annotations are optional.
PubTator relation annotations are ignored. `--limit` defaults to 1 document.
Use a new output directory for each run. Model names, server URL and timeout
are configurable; see `python kg_construction/build.py --help`.

## Pipeline

1. Read title, abstract, PubTator entity IDs and PMID metadata.
2. Extract typed entities and supplement them with PubTator annotations.
3. Extract relations and retry for isolated entities.
4. Merge entities by identifier, deduplicate relations and remove isolated nodes.
5. Export per-document graphs and entity/document lookups.

Prompts and schemas are in `itext2kg/`. Defaults: entity/relation thresholds
0.9/0.4, name/type embedding weights 0.6/0.4, 5 extraction attempts and
3 isolated-entity correction rounds. Each document uses one section, so the
cross-section entity similarity merge is not exercised.

## Outputs

- `delirium_kg_v2.json` / `.pkl`: per-document entities and relations.
- `entities_doc2kg.json` / `.pkl`: entity-to-document triple lookups.
- `chunk_index.json` / `.pkl`: entity-to-document text mappings.
- `<PMID>.trace.json` and `manifest.json`: model responses and run settings.

The graph includes article nodes and `reported` provenance links. These files
use the application's KG schemas; RAG vector indexes are not generated.

## Offline example and tests

The recorded DeepSeek-R1 trace is provided only as an offline regression fixture
for testing the construction pipeline; the KG construction configuration used
in the study employed Gemma3-27B.

```sh
python kg_construction/build.py \
  --input kg_construction/examples/22815080.txt \
  --replay kg_construction/examples/22815080.trace.json \
  --output-dir /tmp/pharmkgpt-kg-replay

python -m unittest discover -s tests -p 'test_kg_construction.py'
```

See [SOURCE_NOTES.md](SOURCE_NOTES.md) for attribution and implementation changes.
