import os
from app.src.engine.core.providers.embedding_provider import HFEmbeddingProvider
from app.src.engine.core.providers.reranker_provider import HFRerankerProvider
from app.src.engine.core.providers.sparse_provider import SparseProvider
import yaml


def load_model_config():
    path = "app/config/model_config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


class ProviderFactory:
    def __init__(self):
        config = load_model_config()

        self.embedding = HFEmbeddingProvider()
        self.reranker = HFRerankerProvider()
        self.sparse = SparseProvider(
            model_name=config["sparse_model"]
        )