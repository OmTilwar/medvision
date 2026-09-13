"""
Unit tests for MedVision Clinical Ragas Evaluation Module with 2-Stage Retrieval & Cross-Encoder.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ragas_clinical_eval import (
    MedVisionRagasEvaluator,
    CLINICAL_GOLD_BENCHMARK,
    clean_tokens,
    expand_clinical_query
)

class TestMedVisionRagas:
    @pytest.fixture(scope="class")
    def evaluator(self):
        return MedVisionRagasEvaluator()

    def test_clean_tokens(self):
        tokens = clean_tokens("Pneumonia bilateral lower-lobe consolidation")
        assert "pneumonia" in tokens
        assert "consolidation" in tokens

    def test_expand_clinical_query(self):
        exp = expand_clinical_query("What are the primary findings?")
        assert "findings" in exp or "pathology" in exp

    def test_context_precision(self, evaluator):
        retrieved = [(2, "Findings chunk", 0.95), (0, "Header chunk", 0.40)]
        precision = evaluator.compute_context_precision(retrieved, target_idx=2)
        assert precision == 1.0

    def test_context_recall(self, evaluator):
        ctx = "Bilateral lower lobe patchy consolidation consistent with bacterial pneumonia. No pleural effusion."
        facts = ["bilateral lower lobe", "patchy consolidation", "bacterial pneumonia", "no pleural effusion"]
        recall = evaluator.compute_context_recall(ctx, facts)
        assert recall == 1.0

    def test_clinical_faithfulness_grounded(self, evaluator):
        report = "FINDINGS: 4mm subpleural nodule in right upper lobe. No lymphadenopathy."
        response = "A 4mm subpleural nodule in right upper lobe. No lymphadenopathy."
        faith = evaluator.compute_faithfulness(response, report)
        assert faith == 1.0

    def test_clinical_faithfulness_hallucinated(self, evaluator):
        report = "FINDINGS: 4mm subpleural nodule in right upper lobe. No lymphadenopathy."
        response = "12mm mass in left hemithorax with extensive effusion."  # Phantom pathologies
        faith = evaluator.compute_faithfulness(response, report)
        assert faith == 0.0

    def test_retrieve_naive(self, evaluator):
        chunks = [
            "PATIENT ID: 12345",
            "FINDINGS: Bilateral lower lobe patchy consolidation consistent with bacterial pneumonia.",
            "IMPRESSION: Bacterial pneumonia."
        ]
        res = evaluator.retrieve_naive("What are findings?", chunks, top_k=2)
        assert len(res) == 2

    def test_retrieve_hybrid_rrf(self, evaluator):
        chunks = [
            "PATIENT ID: 12345",
            "FINDINGS: Bilateral lower lobe patchy consolidation consistent with bacterial pneumonia.",
            "IMPRESSION: Bacterial pneumonia."
        ]
        res = evaluator.retrieve_hybrid_rrf("What are findings?", chunks, top_k=2)
        assert len(res) == 2
        assert any(r[0] == 1 for r in res)

    def test_retrieve_cross_encoder_rerank(self, evaluator):
        chunks = [
            "PATIENT ID: 12345",
            "FINDINGS: Bilateral lower lobe patchy consolidation consistent with bacterial pneumonia.",
            "IMPRESSION: Bacterial pneumonia."
        ]
        res = evaluator.retrieve_cross_encoder_rerank("What are primary radiological findings?", chunks, top_k=2)
        assert len(res) == 2

    def test_full_evaluation_run(self, evaluator):
        res = evaluator.evaluate_model("medvision", retrieval_strategy="cross_encoder")
        assert "context_precision" in res
        assert "context_recall" in res
        assert "faithfulness" in res
        assert "answer_relevancy" in res
        assert res["faithfulness"] >= 0.90
