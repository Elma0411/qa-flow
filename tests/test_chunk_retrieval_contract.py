import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.integrated_pipeline.service import _apply_image_replacements_to_chunks
from app.services.doc_chunks import (
    DOC_TREE_CHUNKS_COLLECTION,
    DOC_TREE_CHUNKS_SCHEMA_VERSION,
    DocumentChunkStore,
)
from app.routers.doc_chunks import (
    ChunkRebuildRequest,
    get_document_assets,
    qa_by_chunk,
    rebuild_chunks,
)
from app.services.debug.qa_store import upsert_qa_debug_items
from qa.chunking.easy_dataset import build_tree_chunks_easy_dataset
from qa.retrieval import (
    BM25Index,
    EvidenceChunk,
    EvidenceWindowBuilder,
    reciprocal_rank_fusion,
)


class ChunkRetrievalContractTests(unittest.TestCase):
    def _chunk(self, markdown: str, *, chunk_size: int = 120):
        return build_tree_chunks_easy_dataset(
            markdown,
            chunk_size=chunk_size,
            original_filename="contract.md",
            task_id="contract-task",
            doc_id="contract-doc",
            split_type="markdown",
        )

    def test_internal_section_can_own_content(self):
        _texts, chunks, _report = self._chunk(
            "# 文档\n\n总则正文。\n\n## 子章节\n\n子章节正文。"
        )

        parent = next(chunk for chunk in chunks if "总则正文" in chunk["text"])
        child = next(chunk for chunk in chunks if "子章节正文" in chunk["text"])
        self.assertFalse(parent["section_is_leaf"])
        self.assertTrue(child["section_is_leaf"])
        self.assertEqual(parent["section_path"], child["section_parent_path"])
        self.assertNotEqual(parent["chunk_id"], child["chunk_id"])

    def test_empty_parent_does_not_create_content_chunk(self):
        _texts, chunks, _report = self._chunk(
            "# 文档\n\n## 空父章节\n\n### 子章节\n\n唯一正文。"
        )

        self.assertEqual(1, len(chunks))
        self.assertEqual("文档>空父章节>子章节", chunks[0]["title_path"])
        self.assertEqual(3, chunks[0]["section_level"])

    def test_long_section_uses_fragments_without_fake_part_section(self):
        body = "第一句内容。" * 60
        _texts, chunks, _report = self._chunk(
            f"# 文档\n\n## 很长章节\n\n{body}",
            chunk_size=90,
        )
        section_chunks = [chunk for chunk in chunks if chunk["title_path"] == "文档>很长章节"]

        self.assertGreater(len(section_chunks), 1)
        self.assertEqual(1, len({chunk["section_path"] for chunk in section_chunks}))
        self.assertEqual(1, len({chunk["fragment_group_id"] for chunk in section_chunks}))
        self.assertEqual(
            list(range(1, len(section_chunks) + 1)),
            [chunk["fragment_index"] for chunk in section_chunks],
        )
        self.assertTrue(all(chunk["fragment_count"] == len(section_chunks) for chunk in section_chunks))
        self.assertTrue(all("Part " not in chunk["title_path"] for chunk in section_chunks))

    def test_image_metadata_belongs_only_to_containing_chunk(self):
        chunks = [
            {
                "chunk_id": "c1",
                "text": "前文 [[IMAGE_REF:img_1]] 后文",
                "text_for_embedding": "前文 [[IMAGE_REF:img_1]] 后文",
            },
            {
                "chunk_id": "c2",
                "text": "普通正文",
                "text_for_embedding": "普通正文",
            },
        ]
        placement_details = [
            {"image_id": "img_1", "accepted": True, "score": 0.9},
            {"image_id": "img_2", "accepted": False, "score": 0.2},
        ]

        final_chunks = _apply_image_replacements_to_chunks(
            chunks,
            {"img_1": "图片事实"},
            placement_details,
        )

        self.assertEqual(["img_1"], chunks[0]["source_asset_ids"])
        self.assertEqual("mixed", chunks[0]["content_kind"])
        self.assertEqual(["img_1"], chunks[0]["image_replacements"]["accepted_ids"])
        self.assertEqual(1, len(chunks[0]["image_replacements"]["placement_details"]))
        self.assertEqual([], chunks[1]["source_asset_ids"])
        self.assertEqual("text", chunks[1]["content_kind"])
        self.assertEqual([], chunks[1]["image_replacements"]["accepted_ids"])
        self.assertEqual([], chunks[1]["image_replacements"]["placement_details"])
        self.assertIn("图片事实", final_chunks[0])

    def test_heading_plus_single_image_is_an_image_description_chunk(self):
        chunks = [
            {
                "chunk_id": "c1",
                "text": "## 检查示意图\n[[IMAGE_REF:img_1]]",
                "text_for_embedding": "文档\n## 检查示意图\n[[IMAGE_REF:img_1]]",
            }
        ]

        _apply_image_replacements_to_chunks(
            chunks,
            {"img_1": "图片展示检查位置。"},
            [{"image_id": "img_1", "accepted": True, "score": 0.9}],
        )

        self.assertEqual("image_description", chunks[0]["content_kind"])
        self.assertEqual(["img_1"], chunks[0]["source_asset_ids"])

    def test_chunk_id_uses_section_chunk_index(self):
        _texts, chunks, _report = self._chunk(
            "# 文档\n\n## 重复内容\n\n相同。\n\n相同。",
            chunk_size=4,
        )
        repeated = [chunk for chunk in chunks if "相同" in chunk["text"]]
        self.assertEqual(len(repeated), len({chunk["chunk_id"] for chunk in repeated}))

    def test_storage_uses_versioned_v2_collection(self):
        self.assertEqual("doc_content_chunks_v2", DOC_TREE_CHUNKS_COLLECTION)
        self.assertEqual(2, DOC_TREE_CHUNKS_SCHEMA_VERSION)

    def test_document_chunk_collection_name_is_not_configurable_to_legacy_schema(self):
        from app.services.doc_chunks import service

        self.assertFalse(hasattr(service, "LEGACY_DOC_TREE_CHUNKS_COLLECTION"))
        self.assertNotIn("doc_tree_chunks_v2_collection", service.__dict__)

    def test_repeated_sibling_titles_get_distinct_section_paths(self):
        _texts, chunks, _report = self._chunk(
            "# 文档\n\n## 要求\n\n第一处要求。\n\n## 要求\n\n第二处要求。"
        )

        repeated = [chunk for chunk in chunks if chunk["title_path"] == "文档>要求"]
        self.assertEqual(2, len(repeated))
        self.assertNotEqual(repeated[0]["section_path"], repeated[1]["section_path"])

    def test_rebuild_validation_and_embedding_finish_before_initialization(self):
        store = DocumentChunkStore("test_v2")
        valid = {
            "chunk_id": "c1",
            "doc_id": "d1",
            "task_id": "t1",
            "original_filename": "x.md",
            "chunk_index": 1,
            "section_chunk_index": 1,
            "section_path": "1",
            "section_parent_path": "",
            "section_level": 1,
            "section_is_leaf": True,
            "fragment_group_id": "g1",
            "fragment_index": 1,
            "fragment_count": 1,
            "content_kind": "text",
            "source_asset_ids": [],
            "text": "正文",
            "text_for_embedding": "正文",
        }

        with patch.object(store, "_attach_embeddings", side_effect=RuntimeError("embed failed")), patch.object(
            store,
            "ensure_initialized",
        ) as ensure:
            with self.assertRaisesRegex(RuntimeError, "embed failed"):
                store.rebuild([valid], task_id="t1")
            ensure.assert_not_called()

        with patch.object(store, "_attach_embeddings") as embed, patch.object(
            store,
            "ensure_initialized",
        ) as ensure:
            with self.assertRaisesRegex(ValueError, "missing"):
                store.rebuild([{**valid, "section_path": ""}], task_id="t1")
            embed.assert_not_called()
            ensure.assert_not_called()

    def test_rebuild_endpoint_regenerates_v2_metadata_from_source_text(self):
        payload = ChunkRebuildRequest(
            task_id="rebuild-task",
            original_filename="source.md",
            text="# 指南\n\n## 材料\n\n提交身份证明。",
            chunk_size=120,
            split_type="markdown",
        )

        with patch("app.routers.doc_chunks.rebuild_doc_tree_chunks") as rebuild:
            rebuild.return_value = {"success": True, "stored_count": 1}
            result = asyncio.run(rebuild_chunks(payload))

        chunks = rebuild.call_args.args[0]
        self.assertEqual("rebuild-task", rebuild.call_args.kwargs["task_id"])
        self.assertTrue(chunks)
        self.assertEqual("1.1", chunks[0]["section_path"])
        self.assertEqual("text", chunks[0]["content_kind"])
        self.assertEqual([], chunks[0]["source_asset_ids"])
        self.assertEqual("source.md", result["original_filename"])

    def test_rebuild_endpoint_surfaces_storage_failure(self):
        from fastapi import HTTPException

        payload = ChunkRebuildRequest(
            task_id="rebuild-task",
            original_filename="source.md",
            text="# 指南\n\n正文。",
        )
        with patch("app.routers.doc_chunks.rebuild_doc_tree_chunks") as rebuild:
            rebuild.return_value = {"success": False, "message": "milvus unavailable"}
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(rebuild_chunks(payload))

        self.assertEqual(503, raised.exception.status_code)

    def test_document_assets_returns_complete_text_and_all_qa_pages(self):
        chunks = [
            {
                "chunk_id": "c2",
                "doc_id": "doc-1",
                "task_id": "task-1",
                "original_filename": "source.md",
                "chunk_index": 2,
                "text": "第二段。",
            },
            {
                "chunk_id": "c1",
                "doc_id": "doc-1",
                "task_id": "task-1",
                "original_filename": "source.md",
                "chunk_index": 1,
                "text": "第一段。",
            },
        ]

        def list_items(**kwargs):
            page = kwargs["page"]
            return {
                "items": [{"id": f"qa-{page}"}],
                "pagination": {"total_items": 2, "total_pages": 2},
            }

        with patch("app.routers.doc_chunks.fetch_chunks_by_doc_id") as fetch, patch(
            "app.routers.doc_chunks.admin_qa_service.list_qa_items",
            side_effect=list_items,
        ) as list_qa, patch("app.routers.doc_chunks.get_debug_map") as debug_map:
            fetch.return_value = {"success": True, "chunks": chunks}
            debug_map.side_effect = lambda ids: {
                qa_id: {"qa_generation_unit_mode": "point"} for qa_id in ids
            }
            result = asyncio.run(
                get_document_assets(
                    doc_id="doc-1",
                    task_id="task-1",
                    original_filename="source.md",
                    include_full_text=True,
                    include_qas=True,
                    include_chunks=True,
                    qa_only_active=True,
                    qa_page_size=1,
                )
            )

        self.assertEqual("第一段。\n\n第二段。", result["full_text"])
        self.assertEqual(["qa-1", "qa-2"], [item["id"] for item in result["qas"]])
        self.assertTrue(all(item["qa_generation_unit_mode"] == "point" for item in result["qas"]))
        self.assertEqual(2, result["total_chunks"])
        self.assertEqual("doc_content_chunks_v2", result["collection_name"])
        self.assertEqual([1, 2], [call.kwargs["page"] for call in list_qa.call_args_list])

    def test_chunk_qa_includes_secondary_primary_sources_from_debug_store(self):
        qa_item = {
            "id": "qa-summary-1",
            "task_id": "task-multi-source",
            "original_filename": "source.md",
            "source": "c1",
            "source_chunk_id": "c1",
            "source_chunk_ids": ["c1", "c2"],
            "question": "办理需要哪些材料和多长时间？",
            "answer": "需提交申请表，五个工作日内办结。",
            "filtered": False,
            "is_primary": True,
        }
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
            "os.environ",
            {"QA_DEBUG_DB_PATH": str(Path(tmp_dir) / "qa-debug.sqlite3")},
        ):
            upsert_qa_debug_items([qa_item])
            with patch("app.routers.doc_chunks.milvus_service.MILVUS_AVAILABLE", False), patch(
                "app.routers.doc_chunks.milvus_service.milvus_client", None
            ), patch("app.routers.doc_chunks.get_chunk_by_id", return_value={"success": False}):
                result = asyncio.run(qa_by_chunk("c2", page=1, page_size=20))

        self.assertEqual("debug", result["source"])
        self.assertEqual(["qa-summary-1"], [item["id"] for item in result["items"]])


class RetrievalContractTests(unittest.TestCase):
    def _chunks(self):
        return [
            EvidenceChunk(
                chunk_id="c1",
                chunk_index=1,
                section_path="1.1",
                section_parent_path="1",
                section_level=2,
                section_is_leaf=False,
                section_chunk_index=1,
                title_path="文档>父章节",
                fragment_group_id="g1",
                fragment_index=1,
                fragment_count=1,
                content_kind="text",
                source_asset_ids=(),
                text="父章节给出办理条件。",
                retrieval_text="父章节 办理条件",
            ),
            EvidenceChunk(
                chunk_id="c2",
                chunk_index=2,
                section_path="1.1.1",
                section_parent_path="1.1",
                section_level=3,
                section_is_leaf=True,
                section_chunk_index=1,
                title_path="文档>父章节>材料",
                fragment_group_id="g2",
                fragment_index=1,
                fragment_count=2,
                content_kind="text",
                source_asset_ids=(),
                text="申请材料包括身份证明。",
                retrieval_text="申请材料 身份证明",
            ),
            EvidenceChunk(
                chunk_id="c3",
                chunk_index=3,
                section_path="1.1.1",
                section_parent_path="1.1",
                section_level=3,
                section_is_leaf=True,
                section_chunk_index=2,
                title_path="文档>父章节>材料",
                fragment_group_id="g2",
                fragment_index=2,
                fragment_count=2,
                content_kind="text",
                source_asset_ids=(),
                text="还需要提交申请表。",
                retrieval_text="申请材料 申请表",
            ),
            EvidenceChunk(
                chunk_id="c4",
                chunk_index=4,
                section_path="1.1.2",
                section_parent_path="1.1",
                section_level=3,
                section_is_leaf=True,
                section_chunk_index=1,
                title_path="文档>父章节>时限",
                fragment_group_id="g3",
                fragment_index=1,
                fragment_count=1,
                content_kind="text",
                source_asset_ids=(),
                text="办理时限为五个工作日。",
                retrieval_text="办理时限 五个工作日",
            ),
        ]

    def test_standard_bm25_prefers_matching_document(self):
        bm25 = BM25Index(self._chunks())

        ranked = bm25.search("身份证明材料", top_k=4)

        self.assertEqual("c2", ranked[0].chunk_id)
        self.assertGreater(ranked[0].score, ranked[-1].score)

    def test_rrf_uses_ranks_and_deduplicates_chunk_ids(self):
        fused = reciprocal_rank_fusion(
            [["c1", "c2", "c3"], ["c2", "c3", "c4"]],
            rank_constant=60,
        )

        self.assertEqual("c2", fused[0].chunk_id)
        self.assertEqual(4, len(fused))
        self.assertEqual(2, fused[0].source_count)

    def test_fragment_group_is_always_restored(self):
        windows = EvidenceWindowBuilder(self._chunks()).build(
            query="申请需要提交哪些材料",
            ranked_chunk_ids=["c2"],
        )

        self.assertIn("c2", windows[0].chunk_ids)
        self.assertIn("c3", windows[0].chunk_ids)
        self.assertEqual("fragment_group", windows[0].reason)

    def test_multi_child_match_emits_parent_body_variants(self):
        windows = EvidenceWindowBuilder(self._chunks()).build(
            query="父章节有哪些材料和办理时限",
            ranked_chunk_ids=["c2", "c4"],
        )

        chunk_sets = {tuple(window.chunk_ids) for window in windows}
        self.assertIn(("c2", "c3", "c4"), chunk_sets)
        self.assertIn(("c1", "c2", "c3", "c4"), chunk_sets)

    def test_same_section_only_merges_consecutive_hits(self):
        chunks = self._chunks() + [
            EvidenceChunk(
                chunk_id="c5",
                chunk_index=5,
                section_path="1.1.1",
                section_parent_path="1.1",
                section_level=3,
                section_is_leaf=True,
                section_chunk_index=4,
                title_path="文档>父章节>材料",
                fragment_group_id="g5",
                fragment_index=1,
                fragment_count=1,
                content_kind="text",
                source_asset_ids=(),
                text="第四块材料说明。",
                retrieval_text="第四块材料说明",
            )
        ]

        windows = EvidenceWindowBuilder(chunks).build(
            query="材料",
            ranked_chunk_ids=["c2", "c5"],
        )

        self.assertNotIn(
            ("c2", "c3", "c5"),
            {tuple(window.chunk_ids) for window in windows},
        )

    def test_explicit_dependency_signal_pulls_one_neighbor(self):
        chunks = [
            EvidenceChunk("p", 1, "1", "", 1, True, 1, "章节", "gp", 1, 1, "text", (), "办理条件如下：", "办理条件如下"),
            EvidenceChunk("a", 2, "1", "", 1, True, 2, "章节", "ga", 1, 1, "text", (), "上述条件满足后提交申请。", "提交申请"),
            EvidenceChunk("n", 3, "1", "", 1, True, 3, "章节", "gn", 1, 1, "text", (), "无关后文。", "无关后文"),
        ]

        windows = EvidenceWindowBuilder(chunks).build(
            query="如何提交",
            ranked_chunk_ids=["a"],
        )

        dependency = next(window for window in windows if window.reason == "dependency_context")
        self.assertEqual(("p", "a"), dependency.chunk_ids)

    def test_forward_dependency_signal_pulls_following_neighbor_only(self):
        chunks = [
            EvidenceChunk("p", 1, "1", "", 1, True, 1, "章节", "gp", 1, 1, "text", (), "前文。", "前文"),
            EvidenceChunk("a", 2, "1", "", 1, True, 2, "章节", "ga", 1, 1, "text", (), "办理条件如下：", "办理条件"),
            EvidenceChunk("n", 3, "1", "", 1, True, 3, "章节", "gn", 1, 1, "text", (), "年满十八周岁。", "十八周岁"),
        ]

        windows = EvidenceWindowBuilder(chunks).build(
            query="办理条件",
            ranked_chunk_ids=["a"],
        )

        dependency = next(window for window in windows if window.reason == "dependency_context")
        self.assertEqual(("a", "n"), dependency.chunk_ids)


if __name__ == "__main__":
    unittest.main()
