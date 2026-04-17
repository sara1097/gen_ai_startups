import pytest
from unittest.mock import patch, MagicMock
from app.src.engine.core.providers.sparse_provider import SparseProvider


class TestSparseProvider:

    @patch("app.src.engine.core.providers.sparse_provider.SparseTextEmbedding")
    def test_encode_returns_object_with_indices(self, mock_model_class):
        """الـ encode لازم يرجع object فيه indices"""
        # جهز الـ sparse vector الوهمي
        mock_sparse_vec = MagicMock()
        mock_sparse_vec.indices = [1, 5, 10, 20]
        mock_sparse_vec.values  = [0.3, 0.5, 0.1, 0.8]

        mock_model_instance = MagicMock()
        # .embed() بترجع generator، bنحاكيه بـ iter
        mock_model_instance.embed.return_value = iter([mock_sparse_vec])
        mock_model_class.return_value = mock_model_instance

        provider = SparseProvider(model_name="prithivida/Splade_PP_en_v1")
        result = provider.encode("payment startup")

        assert hasattr(result, "indices")
        assert hasattr(result, "values")

    @patch("app.src.engine.core.providers.sparse_provider.SparseTextEmbedding")
    def test_encode_indices_and_values_same_length(self, mock_model_class):
        """طول الـ indices لازم يساوي طول الـ values"""
        mock_sparse_vec = MagicMock()
        mock_sparse_vec.indices = [1, 5, 10]
        mock_sparse_vec.values  = [0.3, 0.5, 0.1]

        mock_model_instance = MagicMock()
        mock_model_instance.embed.return_value = iter([mock_sparse_vec])
        mock_model_class.return_value = mock_model_instance

        provider = SparseProvider(model_name="prithivida/Splade_PP_en_v1")
        result = provider.encode("test")

        assert len(result.indices) == len(result.values)

    @patch("app.src.engine.core.providers.sparse_provider.SparseTextEmbedding")
    def test_encode_called_with_list(self, mock_model_class):
        """الـ model لازم يتنادى بـ list مش string"""
        mock_sparse_vec = MagicMock()
        mock_sparse_vec.indices = [1]
        mock_sparse_vec.values  = [0.9]

        mock_model_instance = MagicMock()
        mock_model_instance.embed.return_value = iter([mock_sparse_vec])
        mock_model_class.return_value = mock_model_instance

        provider = SparseProvider(model_name="prithivida/Splade_PP_en_v1")
        provider.encode("hello")

        # تأكد إن .embed اتنادت بـ list
        mock_model_instance.embed.assert_called_once_with(["hello"])