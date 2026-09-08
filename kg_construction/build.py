"""Build PharmKGPT document KGs using the bundled project iText2KG implementation."""

import argparse
import hashlib
import importlib.metadata
import json
import pickle
import platform
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
# Prefer the bundled project fork over a separately installed upstream iText2KG.
sys.path.insert(0, str(HERE))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_pubtator(path):
    """Require one PMID per file, with PubTator title/abstract headers."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path}: missing PubTator title/abstract headers")
    title, abstract = lines[0].split("|", 2), lines[1].split("|", 2)
    if (len(title) != 3 or len(abstract) != 3 or title[1] != "t"
            or abstract[1] != "a" or not title[0].isdigit()
            or title[0] != abstract[0] or not abstract[2].strip()):
        raise ValueError(f"{path}: expected matching PMID|t|title and PMID|a|abstract")
    for line in lines[2:]:
        if not line.strip():
            continue
        columns = line.split("\t")
        # PubTator relation annotations are present in project inputs. The
        # historical pipeline uses entity annotations and extracts relations anew.
        if len(columns) == 4 and columns[0] == title[0]:
            continue
        if len(columns) not in (5, 6) or columns[0] != title[0]:
            raise ValueError(f"{path}: expected 5/6-column entity annotation for PMID {title[0]}")
        if not columns[1].isdigit() or not columns[2].isdigit():
            raise ValueError(f"{path}: annotation offsets must be integers")
    return title[0], f"Title: {title[2]} Abstract: {abstract[2]}"


class ModelTrace:
    """Record actual model I/O, or replay it through the same construction code."""

    def __init__(self, data=None, llm=None, embeddings=None):
        self.replay = data is not None
        self.data = data if self.replay else {"llm": [], "embeddings": []}
        self.llm = llm
        self.embeddings = embeddings
        self.positions = {"llm": 0, "embeddings": 0}

    def _next(self, kind, request):
        position = self.positions[kind]
        if position >= len(self.data[kind]):
            raise ValueError(f"Replay exhausted at {kind} call {position + 1}")
        item = self.data[kind][position]
        if item["input"] != request:
            raise ValueError(f"Replay input mismatch at {kind} call {position + 1}")
        self.positions[kind] += 1
        return item["output"]

    def invoke(self, prompt):
        from langchain_core.messages import AIMessage
        request = prompt.to_string()
        if self.replay:
            return AIMessage(content=self._next("llm", request))
        message = self.llm.invoke(prompt)
        self.data["llm"].append({"input": request, "output": message.content,
                                 "metadata": message.response_metadata})
        return message

    def embed_documents(self, texts):
        if self.replay:
            return self._next("embeddings", texts)
        result = self.embeddings.embed_documents(texts)
        self.data["embeddings"].append({"input": texts, "output": result})
        return result

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def assert_consumed(self):
        if self.replay:
            for kind, position in self.positions.items():
                if position != len(self.data[kind]):
                    raise ValueError(f"Replay has unused {kind} calls")


def construct_document(path, trace):
    from langchain_core.runnables import RunnableLambda
    from itext2kg import iText2KG
    from itext2kg.utils import PubtatorProcessor

    llm = RunnableLambda(trace.invoke)
    processor = PubtatorProcessor(str(path), llm)
    processor.pubtator_info["abstract"] = {
        "context": processor.block[-1], "source": processor.properties_info["source"],
    }
    graph = iText2KG(llm_model=llm, embeddings_model=trace).build_graph(
        sections=[processor.block], source=processor.properties_info,
        entities_info=processor.pubtator_info, ent_threshold=0.9, rel_threshold=0.4,
        max_tries=5, max_tries_isolated_entities=3,
        entity_name_weight=0.6, entity_label_weight=0.4,
    )
    trace.assert_consumed()
    return {
        "entities": [
            {"name": e.name, "label": e.label, "properties_info": e.properties_info}
            for e in graph.entities
        ],
        "relations": [
            {"startEntity": r.startEntity.name, "endEntity": r.endEntity.name,
             "name": r.name, "properties_info": r.properties_info}
            for r in graph.relationships
        ],
    }


def export_artifacts(documents, texts, output):
    """Export the three KG/lookup schemas read by RAGEngine.load_kg()."""
    doc2kg, chunks, application_graphs = {}, {}, {}
    for pmid, graph in documents.items():
        article = "pmid" + pmid
        triples = [(r["startEntity"], r["name"], r["endEntity"]) for r in graph["relations"]]
        names = {e["name"] for e in graph["entities"]}
        if article in names:
            raise ValueError(f"Entity collides with article ID: {article}")
        application_graphs[article] = {
            "entities": graph["entities"] + [{"name": article, "label": "abstract",
                                               "properties_info": {"context": texts[pmid]}}],
            "relations": graph["relations"] + [
                {"startEntity": article, "endEntity": name, "name": "reported",
                 "properties_info": {"source": "PMID" + pmid}}
                for name in sorted(names)
            ],
        }
        for name in sorted(names):
            triples.append((article, "reported", name))
        # Article nodes have a self lookup; entities link back to each article.
        for name in sorted(names | {article}):
            doc2kg.setdefault(name, {})[article] = [t for t in triples if name in (t[0], t[2])]
            chunks.setdefault(name, {})[article] = article + ": " + texts[pmid]
    values = {
        "delirium_kg_v2": application_graphs,
        "entities_doc2kg": {"ents": sorted(doc2kg), "doc2kg": doc2kg},
        "chunk_index": chunks,
    }
    for name, value in values.items():
        write_json(output / (name + ".json"), value)
        with (output / (name + ".pkl")).open("xb") as handle:
            pickle.dump(value, handle, protocol=4)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="PubTator file or directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="New, non-existing directory")
    parser.add_argument("--limit", type=positive, default=1, help="Number of documents (default: 1)")
    parser.add_argument("--model", default="deepseek-r1:32b")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=positive, default=300)
    parser.add_argument("--reasoning", action=argparse.BooleanOptionalAction, default=None,
                        help="Separate thinking output for models supporting Ollama reasoning")
    parser.add_argument("--replay", type=Path, help="Recorded trace for one matching input document")
    args = parser.parse_args()
    manifest = None
    try:
        paths = sorted(args.input.glob("*.txt")) if args.input.is_dir() else [args.input]
        paths = paths[:args.limit]
        if not paths:
            raise ValueError("No .txt files found")
        inputs = [(path, *validate_pubtator(path)) for path in paths]
        if len({pmid for _, pmid, _ in inputs}) != len(inputs):
            raise ValueError("Duplicate input PMIDs")
        if args.replay and len(inputs) != 1:
            raise ValueError("Replay requires exactly one document")
        if args.output_dir.exists():
            raise ValueError("Output directory exists; choose a new directory")
        import langchain_core  # Check construction dependencies before creating output.
        import itext2kg
        if Path(itext2kg.__file__).resolve().parent != HERE / "itext2kg":
            raise ValueError("The bundled project iText2KG package was not imported")
        versions = {p: importlib.metadata.version(p) for p in (
            "langchain", "langchain-core", "langchain-ollama", "pydantic", "numpy", "scikit-learn",
        )}
        manifest = {
            "status": "running", "mode": "replay" if args.replay else "live",
            "python": platform.python_version(), "dependencies": versions,
            "model": args.model, "embedding_model": args.embedding_model,
            "temperature": 0, "timeout": args.timeout, "reasoning": args.reasoning,
            "parameters": {"ent_threshold": 0.9, "rel_threshold": 0.4,
                           "max_tries": 5, "max_tries_isolated_entities": 3,
                           "entity_name_weight": 0.6, "entity_label_weight": 0.4},
            "code_sha256": {str(p.relative_to(HERE)): sha256(p) for p in sorted(HERE.rglob("*.py"))},
            "inputs": [{"file": path.name, "pmid": pmid, "sha256": sha256(path)}
                       for path, pmid, _ in inputs],
            "documents": [],
        }
        args.output_dir.mkdir(parents=True)
        documents, texts = {}, {}
        for path, pmid, text in inputs:
            if args.replay:
                data = json.loads(args.replay.read_text(encoding="utf-8"))
                if data["input_sha256"] != sha256(path):
                    raise ValueError("Replay input file hash mismatch")
                trace = ModelTrace(data=data)
                manifest["recorded_model"] = data.get("model")
                manifest["recorded_embedding_model"] = data.get("embedding_model")
            else:
                from langchain_ollama import ChatOllama, OllamaEmbeddings
                trace = ModelTrace(
                    llm=ChatOllama(model=args.model, base_url=args.base_url, temperature=0,
                                   reasoning=args.reasoning,
                                   client_kwargs={"timeout": args.timeout}),
                    embeddings=OllamaEmbeddings(model=args.embedding_model, base_url=args.base_url,
                                                client_kwargs={"timeout": args.timeout}),
                )
                trace.data.update(input_sha256=sha256(path), model=args.model,
                                  embedding_model=args.embedding_model)
            print(f"Constructing PMID {pmid} ({manifest['mode']})...", flush=True)
            try:
                documents[pmid] = construct_document(path, trace)
            finally:
                write_json(args.output_dir / (pmid + ".trace.json"), trace.data)
            texts[pmid] = text
            manifest["documents"].append({"pmid": pmid, "entities": len(documents[pmid]["entities"]),
                                          "relations": len(documents[pmid]["relations"])})
        export_artifacts(documents, texts, args.output_dir)
        manifest["status"] = "complete"
        manifest["output_sha256"] = {p.name: sha256(p) for p in sorted(args.output_dir.iterdir())}
        write_json(args.output_dir / "manifest.json", manifest)
    except Exception as exc:
        if manifest is not None and args.output_dir.is_dir():
            manifest.update(status="failed", error=str(exc))
            write_json(args.output_dir / "manifest.json", manifest)
        parser.exit(1, f"KG construction failed: {exc}\n")
    print(f"Saved {len(documents)} document graph(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
