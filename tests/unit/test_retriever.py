import pytest
from unittest.mock import patch, MagicMock


def make_mock_point(name="TestStartup", sector="fintech"):
    """Helper بيعمل mock point جاهز"""
    point = MagicMock()
    point.payload = {
        "name": name,
        "sector": sector,
        "use_case": "payment processing",
        "solution": "AI payments platform",
        "description": "A startup description",
        "domain": "finance",
        "link": "https://example.com"
    }
    return point


class TestRetrieveTopK:

    @patch("app.src.engine.rag.retriver.reranker_provider")
    @patch("app.src.engine.rag.retriver.sparse_provider")
    @patch("app.src.engine.rag.retriver.embedding_provider")
    @patch("app.src.engine.rag.retriver.qdrant_client")
    def test_returns_correct_number_of_results(
        self, mock_qdrant, mock_embed, mock_sparse, mock_reranker
    ):
        """retrieve_topk لازم يرجع k نتايج"""
        # جهز الـ providers
        mock_embed.encode.return_value = [0.1] * 768

        mock_sparse_vec = MagicMock()
        mock_sparse_vec.indices.tolist.return_value = [1, 2, 3]
        mock_sparse_vec.values.tolist.return_value  = [0.3, 0.5, 0.2]
        mock_sparse.encode.return_value = mock_sparse_vec

        mock_reranker.score.return_value = 0.9

        # جهز 10 نقاط مختلفة الأسماء
        points = [make_mock_point(name=f"Startup{i}") for i in range(10)]
        mock_result = MagicMock()
        mock_result.points = points
        mock_qdrant.query_points.return_value = mock_result

        from startups_ai_generator.app.src.engine.rag.retriever import retrieve_topk
        results = retrieve_topk("payment problem", k=5, sector="Finance")

        assert len(results) == 5

    @patch("app.src.engine.rag.retriver.reranker_provider")
    @patch("app.src.engine.rag.retriver.sparse_provider")
    @patch("app.src.engine.rag.retriver.embedding_provider")
    @patch("app.src.engine.rag.retriver.qdrant_client")
    def test_deduplicates_same_name(
        self, mock_qdrant, mock_embed, mock_sparse, mock_reranker
    ):
        """لو في startups بنفس الاسم يشيل التكرار"""
        mock_embed.encode.return_value = [0.1] * 768

        mock_sparse_vec = MagicMock()
        mock_sparse_vec.indices.tolist.return_value = [1]
        mock_sparse_vec.values.tolist.return_value  = [1.0]
        mock_sparse.encode.return_value = mock_sparse_vec

        mock_reranker.score.return_value = 0.8

        # 5 نقاط بنفس الاسم
        points = [make_mock_point(name="SameStartup")] * 5
        mock_result = MagicMock()
        mock_result.points = points
        mock_qdrant.query_points.return_value = mock_result

        from startups_ai_generator.app.src.engine.rag.retriever import retrieve_topk
        results = retrieve_topk("health problem", k=5)

        assert len(results) == 1

    @patch("app.src.engine.rag.retriver.reranker_provider")
    @patch("app.src.engine.rag.retriver.sparse_provider")
    @patch("app.src.engine.rag.retriver.embedding_provider")
    @patch("app.src.engine.rag.retriver.qdrant_client")
    def test_fallback_when_few_results(
        self, mock_qdrant, mock_embed, mock_sparse, mock_reranker
    ):
        """لو النتايج أقل من k يعيد البحث من غير filter"""
        mock_embed.encode.return_value = [0.1] * 768

        mock_sparse_vec = MagicMock()
        mock_sparse_vec.indices.tolist.return_value = [1]
        mock_sparse_vec.values.tolist.return_value  = [1.0]
        mock_sparse.encode.return_value = mock_sparse_vec

        mock_reranker.score.return_value = 0.5

        # المرة الأولى: نتيجة واحدة بس (أقل من k=5)
        # المرة التانية (fallback): 10 نتايج
        few_results = MagicMock()
        few_results.points = [make_mock_point(name="Only1")]

        many_results = MagicMock()
        many_results.points = [make_mock_point(name=f"Startup{i}") for i in range(10)]

        mock_qdrant.query_points.side_effect = [few_results, many_results]

        from startups_ai_generator.app.src.engine.rag.retriever import retrieve_topk
        results = retrieve_topk("problem", k=5, sector="Finance")

        # اتنادت مرتين — مرة بـ filter ومرة من غيره
        assert mock_qdrant.query_points.call_count == 2