import unittest

from qa.retrieval import EvidenceChunk, EvidenceRetrievalPipeline


class _StaticReranker:
    def rank(self, query, pairs):
        del query
        scores = {
            "c1": 0.1,
            "c2": 0.9,
            "c3": 0.8,
            "c4": 0.7,
        }
        ranked = [(identifier, scores.get(identifier, 0.6)) for identifier, _text in pairs]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked


class _TieReranker:
    def rank(self, query, pairs):
        del query
        pairs = list(pairs)
        if pairs and all(str(identifier).startswith("c") for identifier, _text in pairs):
            atomic_scores = {"c1": 0.1, "c2": 1.0, "c3": 1.0}
            return sorted(
                [(identifier, atomic_scores.get(identifier, 0.0)) for identifier, _text in pairs],
                key=lambda item: (-item[1], item[0]),
            )
        scored = []
        for identifier, text in pairs:
            score = 2.0 if "身份证明" in text and "五个工作日" in text else 1.0
            scored.append((identifier, score))
        return list(reversed(scored))


class RetrievalPipelineTests(unittest.TestCase):
    def test_pipeline_exposes_fixed_stage_trace_and_windows(self):
        chunks = [
            EvidenceChunk("c1", 1, "1", "", 1, False, 1, "文档", "g1", 1, 1, "text", (), "总则", "总则"),
            EvidenceChunk("c2", 2, "1.1", "1", 2, True, 1, "文档>材料", "g2", 1, 2, "text", (), "身份证明", "申请材料 身份证明"),
            EvidenceChunk("c3", 3, "1.1", "1", 2, True, 2, "文档>材料", "g2", 2, 2, "text", (), "申请表", "申请材料 申请表"),
            EvidenceChunk("c4", 4, "1.2", "1", 2, True, 1, "文档>时限", "g3", 1, 1, "text", (), "五个工作日", "办理时限 五个工作日"),
        ]
        pipeline = EvidenceRetrievalPipeline(
            chunks,
            [[1.0, 0.0], [0.8, 0.2], [0.7, 0.3], [0.2, 0.8]],
            reranker=_StaticReranker(),
        )

        result = pipeline.retrieve(
            "申请材料有哪些？",
            [0.9, 0.1],
            final_evidence_k=2,
            evidence_token_budget=1000,
            source_chunk_ids=["c1"],
        )

        self.assertEqual("bm25_dense_rrf_bge_structure_v1", result["trace"]["pipeline"])
        self.assertTrue(result["trace"]["dense_hits"])
        self.assertTrue(result["trace"]["bm25_hits"])
        self.assertTrue(result["trace"]["rrf_hits"])
        self.assertEqual("c2", result["trace"]["atomic_rerank"][0]["chunk_id"])
        self.assertIn("c2", result["selected_chunk_ids"])
        self.assertIn("c3", result["selected_chunk_ids"])
        self.assertEqual(len(result["selected_chunk_ids"]), len(set(result["selected_chunk_ids"])))

    def test_zero_final_evidence_k_selects_no_window(self):
        chunks = [
            EvidenceChunk("c1", 1, "1", "", 1, True, 1, "文档", "g1", 1, 1, "text", (), "总则", "总则"),
            EvidenceChunk("c2", 2, "2", "", 1, True, 1, "材料", "g2", 1, 1, "text", (), "身份证明", "申请材料 身份证明"),
        ]
        pipeline = EvidenceRetrievalPipeline(chunks, [[1.0], [0.8]], reranker=_StaticReranker())

        result = pipeline.retrieve(
            "申请材料",
            [1.0],
            final_evidence_k=0,
            evidence_token_budget=4000,
            source_chunk_ids=["c1"],
        )

        self.assertEqual([], result["selected_windows"])
        self.assertEqual([], result["selected_chunk_ids"])

    def test_single_window_must_fit_token_budget(self):
        chunks = [
            EvidenceChunk("c1", 1, "1", "", 1, True, 1, "文档", "g1", 1, 1, "text", (), "总则", "总则"),
            EvidenceChunk("c2", 2, "2", "", 1, True, 1, "材料", "g2", 1, 1, "text", (), "身份证明" * 100, "申请材料 身份证明"),
        ]
        pipeline = EvidenceRetrievalPipeline(chunks, [[1.0], [0.8]], reranker=_StaticReranker())

        result = pipeline.retrieve(
            "申请材料",
            [1.0],
            final_evidence_k=1,
            evidence_token_budget=2,
            source_chunk_ids=["c1"],
        )

        self.assertEqual([], result["selected_windows"])
        self.assertEqual([], result["selected_chunk_ids"])

    def test_parent_body_variants_are_mutually_exclusive(self):
        chunks = [
            EvidenceChunk("c1", 1, "1", "", 1, False, 1, "文档", "g1", 1, 1, "text", (), "父正文", "父正文"),
            EvidenceChunk("c2", 2, "1.1", "1", 2, True, 1, "文档>材料", "g2", 1, 1, "text", (), "身份证明", "申请材料 身份证明"),
            EvidenceChunk("c4", 4, "1.2", "1", 2, True, 1, "文档>时限", "g4", 1, 1, "text", (), "五个工作日", "办理时限 五个工作日"),
        ]
        pipeline = EvidenceRetrievalPipeline(chunks, [[1.0], [0.9], [0.8]], reranker=_StaticReranker())

        result = pipeline.retrieve(
            "材料和时限",
            [1.0],
            final_evidence_k=5,
            evidence_token_budget=4000,
        )

        sibling_variants = [
            window
            for window in result["selected_windows"]
            if window.reason in {"sibling_hits", "sibling_hits_with_parent_body"}
        ]
        self.assertLessEqual(len(sibling_variants), 1)

    def test_equal_rerank_score_prefers_shorter_window(self):
        chunks = [
            EvidenceChunk("c1", 1, "1", "", 1, False, 1, "文档", "g1", 1, 1, "text", (), "很长的父正文" * 20, "父正文"),
            EvidenceChunk("c2", 2, "1.1", "1", 2, True, 1, "文档>材料", "g2", 1, 1, "text", (), "身份证明", "材料 身份证明"),
            EvidenceChunk("c3", 3, "1.2", "1", 2, True, 1, "文档>时限", "g3", 1, 1, "text", (), "五个工作日", "时限 五个工作日"),
        ]
        pipeline = EvidenceRetrievalPipeline(chunks, [[1.0], [0.9], [0.8]], reranker=_TieReranker())

        result = pipeline.retrieve(
            "材料和时限",
            [1.0],
            final_evidence_k=1,
            evidence_token_budget=4000,
        )

        selected = result["selected_windows"][0]
        self.assertEqual("sibling_hits", selected.reason)
        self.assertNotIn("c1", selected.chunk_ids)


if __name__ == "__main__":
    unittest.main()
