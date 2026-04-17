import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.src.engine.core.providers.embedding_provider import HFEmbeddingProvider


class TestHFEmbeddingProvider:


    @patch("app.src.engine.core.providers.embedding_provider.InferenceClient")
    def test_encode_returns_list(self, mock_client_class):
        
        mock_instance = MagicMock()
        mock_instance.feature_extraction.return_value = [0.1] * 768
        mock_client_class.return_value = mock_instance

        provider = HFEmbeddingProvider()
        result = provider.encode("test input")

        assert isinstance(result, list)

    @patch("app.src.engine.core.providers.embedding_provider.InferenceClient")
    def test_encode_returns_768_dimensions(self, mock_client_class):
      
        mock_instance = MagicMock()
        mock_instance.feature_extraction.return_value = [0.1] * 768
        mock_client_class.return_value = mock_instance

        provider = HFEmbeddingProvider()
        result = provider.encode("fintech startup problem")

        assert len(result) == 768

    @patch("app.src.engine.core.providers.embedding_provider.InferenceClient")
    def test_encode_empty_text_returns_empty_list(self, mock_client_class):
      
        provider = HFEmbeddingProvider()
        result = provider.encode("")

        assert result == []

    @patch("app.src.engine.core.providers.embedding_provider.InferenceClient")
    def test_encode_handles_numpy_array(self, mock_client_class):
        
        mock_instance = MagicMock()
        mock_instance.feature_extraction.return_value = np.array([0.1] * 768)
        mock_client_class.return_value = mock_instance

        provider = HFEmbeddingProvider()
        result = provider.encode("test")

       
        assert isinstance(result, list)
        assert len(result) == 768

    @patch("app.src.engine.core.providers.embedding_provider.InferenceClient")
    def test_encode_handles_nested_list(self, mock_client_class):
        
        mock_instance = MagicMock()
        mock_instance.feature_extraction.return_value = [[0.2] * 768]
        mock_client_class.return_value = mock_instance

        provider = HFEmbeddingProvider()
        result = provider.encode("test")

        assert len(result) == 768
        assert isinstance(result[0], float)