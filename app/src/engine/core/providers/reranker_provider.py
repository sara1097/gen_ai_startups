import os
import json
from huggingface_hub import InferenceClient

class HFRerankerProvider:
    def __init__(self):
        self.client = InferenceClient(token=os.getenv("HF_TOKEN"))
        self.model = "BAAI/bge-reranker-base"

    def score(self, query: str, doc: str) -> float:
        if not query or not doc:
            return 0.0
        try:
            response = self.client.post(
                json={
                    "inputs": {
                        "text": query,
                        "text_pair": doc
                    }
                },
                model=self.model,
            )
            result = json.loads(response)
            # بيرجع list of dicts زي: [{"label": "LABEL_0", "score": 0.98}]
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):  # nested list
                    return float(result[0][0].get("score", 0.0))
                return float(result[0].get("score", 0.0))
        except Exception as e:
            return 0.0
        return 0.0