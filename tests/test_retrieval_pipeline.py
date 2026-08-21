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


class _IrrelevantReranker:
    def rank(self, query, pairs):
        del query
        return [(identifier, -4.0 - index) for index, (identifier, _text) in enumerate(pairs)]


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

        self.assertEqual("bm25_dense_rrf_bge_structure_scope_v3", result["trace"]["pipeline"])
        self.assertEqual(
            len(result["trace"]["selected_windows"]),
            result["trace"]["selected_evidence_window_count"],
        )
        self.assertEqual(
            len(result["selected_chunk_ids"]),
            result["trace"]["selected_evidence_chunk_count"],
        )
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

    def test_irrelevant_candidates_are_not_used_to_fill_final_limit(self):
        chunks = [
            EvidenceChunk("c1", 1, "1", "", 1, True, 1, "主材料", "g1", 1, 1, "text", (), "主材料", "主材料"),
            EvidenceChunk("c2", 2, "2", "", 1, True, 1, "无关条款", "g2", 1, 1, "text", (), "天气晴朗", "天气晴朗"),
        ]
        pipeline = EvidenceRetrievalPipeline(
            chunks,
            [[1.0], [0.8]],
            reranker=_IrrelevantReranker(),
        )

        result = pipeline.retrieve(
            "申请材料有哪些？",
            [1.0],
            final_evidence_k=5,
            evidence_token_budget=4000,
            source_chunk_ids=["c1"],
        )

        self.assertEqual([], result["selected_windows"])
        self.assertEqual([], result["selected_chunk_ids"])
        self.assertEqual(0, result["trace"]["relevance_admission"]["atomic_admitted_count"])

    def test_candidate_far_below_primary_source_is_not_admitted(self):
        chunks = [
            EvidenceChunk("c1", 1, "1", "", 1, True, 1, "主材料", "g1", 1, 1, "text", (), "主材料", "主材料"),
            EvidenceChunk("c2", 2, "1.1", "1", 2, True, 1, "主材料>时限", "g2", 1, 1, "text", (), "五个工作日", "五个工作日"),
        ]

        class CalibratedReranker:
            def rank(self, query, pairs):
                del query
                return [
                    (identifier, 3.0 if identifier == "c1" else 0.1)
                    for identifier, _text in pairs
                ]

        pipeline = EvidenceRetrievalPipeline(chunks, [[1.0], [0.8]], reranker=CalibratedReranker())
        result = pipeline.retrieve(
            "办理需要多长时间？",
            [1.0],
            final_evidence_k=1,
            evidence_token_budget=4000,
            source_chunk_ids=["c1"],
        )

        self.assertEqual([], result["selected_chunk_ids"])
        self.assertEqual(
            "outside_primary_source_band",
            next(
                item["rejection_reason"]
                for item in result["trace"]["atomic_rerank"]
                if item["chunk_id"] == "c2"
            ),
        )

    def test_low_absolute_candidate_close_to_primary_source_is_admitted(self):
        chunks = [
            EvidenceChunk("c1", 1, "1", "", 1, True, 1, "主材料", "g1", 1, 1, "text", (), "五个工作日", "五个工作日"),
            EvidenceChunk("c2", 2, "1.1", "1", 2, True, 1, "主材料>补充时限", "g2", 1, 1, "text", (), "补正时间不计入期限", "补正时间不计入期限"),
        ]

        class LowRelevantReranker:
            def rank(self, query, pairs):
                del query
                return [
                    (identifier, -0.45 if identifier == "c1" else -0.8)
                    for identifier, _text in pairs
                ]

        pipeline = EvidenceRetrievalPipeline(chunks, [[1.0], [0.8]], reranker=LowRelevantReranker())
        result = pipeline.retrieve(
            "办理需要多长时间？",
            [1.0],
            final_evidence_k=1,
            evidence_token_budget=4000,
            source_chunk_ids=["c1"],
        )

        self.assertEqual(["c2"], result["selected_chunk_ids"])

    def test_relevant_sibling_outside_source_structure_scope_is_rejected(self):
        chunks = [
            EvidenceChunk(
                "c1", 1, "1.1", "1", 2, True, 1,
                "文档>经办机构核定", "g1", 1, 1, "text", (),
                "核定时提交工资台账。", "经办机构核定 工资台账",
            ),
            EvidenceChunk(
                "c2", 2, "1.2", "1", 2, True, 1,
                "文档>网上申报上传", "g2", 1, 1, "text", (),
                "网上上传仅需汇总表。", "网上申报 上传 汇总表",
            ),
        ]

        class ScopeReranker:
            def rank(self, query, pairs):
                del query
                return [
                    (identifier, 4.0 if identifier == "c2" else 3.8)
                    for identifier, _text in pairs
                ]

        pipeline = EvidenceRetrievalPipeline(
            chunks,
            [[1.0], [0.9]],
            reranker=ScopeReranker(),
        )
        result = pipeline.retrieve(
            "核定时需要提交哪些资料？",
            [1.0],
            final_evidence_k=2,
            evidence_token_budget=4000,
            source_chunk_ids=["c1"],
        )
        self.assertEqual([], result["selected_chunk_ids"])
        rejected = next(
            item
            for item in result["trace"]["atomic_rerank"]
            if item["chunk_id"] == "c2"
        )
        self.assertEqual("outside_source_structure_scope", rejected["rejection_reason"])


if __name__ == "__main__":
    unittest.main()
