# Source notes

Adapted on 2026-09-08 from the project's `abstract2KG.py`, `AD2KG.py` and
customized `itext2kg` package. Upstream attribution: iText2KG, Auvalab /
Yassir LAIRGI. The supplied license is retained in [LICENSE.itext2kg](LICENSE.itext2kg).

Changes:

- Make model endpoints and file paths configurable; remove unused provider and
  Neo4j imports.
- Fix entity-list parsing, PubTator supplementation, missing properties and
  retention of entities with nonduplicated identifiers.
- Stabilize entity ordering and correct parser error handling.
- Add the CLI, application-format exports and model-response replay.

The original relation prompts, identifier merging and isolated-entity handling
are retained. Relation merging keeps one predicate per ordered endpoint pair;
reverse relations with the same predicate are also deduplicated. The fixes can
change outputs relative to the original scripts.
