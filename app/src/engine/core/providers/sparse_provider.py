from fastembed import SparseTextEmbedding


class SparseProvider:
    """
    Local sparse model (خفيف ومش محتاج سيرفر)
    """

    def __init__(self, model_name: str):
        self.model = SparseTextEmbedding(model_name=model_name)

    def encode(self, text: str):
        """
        Returns sparse vector (indices + values)
        """
        result = list(self.model.embed([text]))[0]
        return result