import os
from huggingface_hub import InferenceClient


class HFEmbeddingProvider:
    """
    Remote embedding model (no local download)
    """

    def __init__(self):
        self.client = InferenceClient(token=os.getenv("HF_TOKEN"))
        self.model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    def encode(self, text: str):
        if not text:
            return []

        result = self.client.feature_extraction(
            model=self.model,
            text=text
        )

        # 🧠 الحل الصح
        import numpy as np

        # لو numpy array
        if isinstance(result, np.ndarray):
            return result.tolist()

        # لو nested list
        if isinstance(result, list) and isinstance(result[0], list):
            return result[0]

        # لو list عادي
        if isinstance(result, list):
            return result

        # fallback
        return list(result)