"""Regression checks for the public construction entry point and project fork."""

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kg_construction"))
from build import ModelTrace, construct_document, export_artifacts, validate_pubtator
from itext2kg.models import Entity, Relationship
from itext2kg.utils import Matcher, PubtatorProcessor


class ConstructionTests(unittest.TestCase):
    def test_recorded_real_sample_rebuilds_reference_graph(self):
        examples = ROOT / "kg_construction/examples"
        trace = ModelTrace(data=json.loads((examples / "22815080.trace.json").read_text()))
        graph = construct_document(examples / "22815080.txt", trace)
        _, text = validate_pubtator(examples / "22815080.txt")
        with tempfile.TemporaryDirectory() as directory:
            export_artifacts({"22815080": graph}, {"22815080": text}, Path(directory))
            actual = json.loads((Path(directory) / "delirium_kg_v2.json").read_text())
        expected = json.loads((examples / "22815080.expected_kg.json").read_text())
        self.assertEqual(actual, expected)

    def test_project_pubtator_input_with_relation_annotations(self):
        pmid, text = validate_pubtator(ROOT / "kg_construction/examples/22815080.txt")
        self.assertEqual(pmid, "22815080")
        self.assertIn("APOE genotype", text)

    def test_distilled_entities_filtered_and_pubtator_supplemented(self):
        processor = PubtatorProcessor.__new__(PubtatorProcessor)
        processor.context = "APOE and Alzheimer's disease"
        processor.pubtator_distilled = {"gene": [{"gene": "APOE"}],
                                       "disease": [{"disease": "Alzheimer's disease"}]}
        distilled = processor._match_distilled_to_context({"entities": [
            {"name": "APOE", "label": "gene"},
            {"name": "Invented gene", "label": "gene"},
        ]})
        result = processor._add_missing_entities(distilled)
        self.assertEqual(result, {"gene": [{"gene": "APOE"}],
                                  "disease": [{"disease": "Alzheimer's disease"}]})

    def test_identifier_merge_retains_singletons_and_rewires_aliases(self):
        gene = Entity(name="apoe", label="gene", properties_info={"unique_id": "Gene ID:348"})
        disease = Entity(name="ad", label="disease", properties_info={"unique_id": "MESH:D000544"})
        alias = Entity(name="alzheimer disease", label="disease", properties_info={"unique_id": "MESH:D000544"})
        relation = Relationship(startEntity=gene, endEntity=disease, name="associated_with")
        entities, relations = Matcher().merge_entities_relationship_by_unique_id(
            [gene, disease, alias], [relation])
        self.assertEqual({e.name for e in entities}, {"apoe", "alzheimer disease"})
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0].endEntity.name, "alzheimer disease")

    def test_replay_rejects_different_requests_and_unused_calls(self):
        trace = ModelTrace(data={"llm": [], "embeddings": [{"input": ["apoe"], "output": [[1, 0]]}]})
        with self.assertRaisesRegex(ValueError, "mismatch"):
            trace.embed_documents(["wrong"])
        with self.assertRaisesRegex(ValueError, "unused"):
            trace.assert_consumed()
        self.assertEqual(trace.embed_query("apoe"), [1, 0])
        trace.assert_consumed()

    def test_construction_and_export_use_application_schema(self):
        # Controlled model replies test orchestration, not extraction accuracy.
        from langchain_core.messages import AIMessage
        replies = [
            {"entities": [{"name": "APOE", "label": "gene"}, {"name": "AD", "label": "disease"}]},
            {"entities": [{"name": "APOE", "label": "gene"}, {"name": "AD", "label": "disease"}]},
            {"relationships": [{"startNode": {"name": "APOE", "label": "gene"},
                                "endNode": {"name": "AD", "label": "disease"},
                                "name": "associated_with"}]},
        ]

        class FakeLLM:
            def invoke(self, prompt):
                return AIMessage(content=json.dumps(replies.pop(0)))

        class FakeEmbeddings:
            def embed_documents(self, texts):
                return [[1.0, 0.0] for _ in texts]

        trace = ModelTrace(llm=FakeLLM(), embeddings=FakeEmbeddings())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "1.txt"
            source.write_text("1|t|Example\n1|a|APOE is associated with AD.\n"
                              "1\t8\t12\tAPOE\tGene\t348\n"
                              "1\t32\t34\tAD\tDisease\tMESH:D000544\n")
            graph = construct_document(source, trace)
            self.assertEqual(len(graph["relations"]), 1)
            self.assertEqual(graph["relations"][0]["properties_info"]["source"], "PMID1")
            # Replay traverses distillation, extraction, embedding and merging again.
            replayed = construct_document(source, ModelTrace(data=trace.data))
            self.assertEqual(replayed, graph)
            export_artifacts({"1": graph}, {"1": "Example text"}, root)
            with (root / "entities_doc2kg.pkl").open("rb") as handle:
                lookup = pickle.load(handle)
            self.assertIn(("apoe", "associated_with", "ad"), lookup["doc2kg"]["apoe"]["pmid1"])
            self.assertIn(("pmid1", "reported", "apoe"), lookup["doc2kg"]["pmid1"]["pmid1"])
            from src.kgvisual import kg_visualization
            with (root / "delirium_kg_v2.pkl").open("rb") as handle:
                application_kg = pickle.load(handle)
            visual = kg_visualization(["pmid1"], application_kg)
            self.assertEqual(len(visual["nodes"]), 3)
            self.assertEqual(len(visual["edges"]), 3)


if __name__ == "__main__":
    unittest.main()
