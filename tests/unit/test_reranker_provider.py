import pytest
import json
from unittest.mock import patch, MagicMock
from app.src.engine.core.providers.reranker_provider import HFRerankerProvider


class TestHFRerankerProvider:

    @patch("app.src.engine.core.providers.reranker_provider.InferenceClient")
    def test_score_returns_float(self, mock_client_class):
        """الـ score لازم يرجع float"""
        mock_instance = MagicMock()
        # بنحاكي الـ response بتاع HF API
        mock_instance.post.return_value = json.dumps(
            [{"label": "LABEL_0", "score": 0.92}]
        ).encode()
        mock_client_class.return_value = mock_instance

        provider = HFRerankerProvider()
        score = provider.score("payment problem", "fintech startup solution")

        assert isinstance(score, float)

    @patch("app.src.engine.core.providers.reranker_provider.InferenceClient")
    def test_score_between_zero_and_one(self, mock_client_class):
        """الـ score لازم يكون بين 0 و 1"""
        mock_instance = MagicMock()
        mock_instance.post.return_value = json.dumps(
            [{"label": "LABEL_0", "score": 0.75}]
        ).encode()
        mock_client_class.return_value = mock_instance

        provider = HFRerankerProvider()
        score = provider.score("healthcare data", "AI health solution")

        assert 0.0 <= score <= 1.0

    @patch("app.src.engine.core.providers.reranker_provider.InferenceClient")
    def test_score_empty_query_returns_zero(self, mock_client_class):
        """لو الـ query فاضي يرجع 0"""
        provider = HFRerankerProvider()
        assert provider.score("", "some document") == 0.0

    @patch("app.src.engine.core.providers.reranker_provider.InferenceClient")
    def test_score_empty_doc_returns_zero(self, mock_client_class):
        """لو الـ doc فاضي يرجع 0"""
        provider = HFRerankerProvider()
        assert provider.score("some query", "") == 0.0

    @patch("app.src.engine.core.providers.reranker_provider.InferenceClient")
    def test_score_api_failure_returns_zero(self, mock_client_class):
        """لو الـ API فشل يرجع 0 بدل ما يكسر"""
        mock_instance = MagicMock()
        mock_instance.post.side_effect = Exception("Connection error")
        mock_client_class.return_value = mock_instance

        provider = HFRerankerProvider()
        score = provider.score("query", "document")

        assert score == 0.0

    @patch("app.src.engine.core.providers.reranker_provider.InferenceClient")
    def test_score_handles_nested_response(self, mock_client_class):
        """لو الـ response nested list يتعامل معاه صح"""
        mock_instance = MagicMock()
        mock_instance.post.return_value = json.dumps(
            [[{"label": "LABEL_0", "score": 0.88}]]
        ).encode()
        mock_client_class.return_value = mock_instance

        provider = HFRerankerProvider()
        score = provider.score("query", "doc")

        assert score == 0.88