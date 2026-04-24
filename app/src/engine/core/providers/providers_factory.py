# app/src/engine/core/providers/providers_factory.py
import logging
import yaml
from qdrant_client import QdrantClient

from app.config.settings import get_settings
from app.src.engine.core.providers.embedding_provider import HFEmbeddingProvider
from app.src.engine.core.providers.reranker_provider import HFRerankerProvider
from app.src.engine.core.providers.sparse_provider import SparseProvider
from app.src.llm.groq_provider import GroqProvider
from app.src.engine.rag.retriever import StartupRetriever
from app.src.engine.core.intent_classification import IntentClassifier

logger = logging.getLogger(__name__)

class ProviderFactory:
    """
    Dependency Injection Container.
    Instantiates and manages the lifecycle of all external providers and services.
    Ensures singletons are created only when needed, avoiding import-time side effects.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProviderFactory, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
            
        logger.info("Initializing ProviderFactory...")
        self.settings = get_settings()
        
        # Initialize LLM Provider
        self.llm = GroqProvider()
        
        # Initialize Intent Classifier
        self.intent_classifier = IntentClassifier(self.llm)
        
        # Initialize Vector DB Client
        self.qdrant_client = QdrantClient(
            url=self.settings.QDRANT_URL, 
            api_key=self.settings.QDRANT_API_KEY
        )
        
        # Initialize Embedding Providers
        self.embedding = HFEmbeddingProvider()
        self.reranker = HFRerankerProvider()
        self.sparse = SparseProvider(model_name=self.settings.SPARSE_MODEL)
        
        # Load Sector Mappings
        self.sector_mappings = self._load_sector_mappings()
        
        # Initialize Retriever
        self.retriever = StartupRetriever(
            qdrant_client=self.qdrant_client,
            embedding_provider=self.embedding,
            sparse_provider=self.sparse,
            reranker_provider=self.reranker,
            sector_mappings=self.sector_mappings
        )
        
        self._initialized = True
        logger.info("ProviderFactory initialized successfully.")

    def _load_sector_mappings(self) -> tuple:
        """Load domain mappings from YAML config."""
        path = "app/config/domain_mapping.yaml"
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                
            return (
                data.get("STARTUP_SECTOR_GROUPS", {}),
                data.get("PROBLEM_TO_STARTUP_GROUPS", {}),
                data.get("BOILERPLATE_SIGNALS", [])
            )
        except Exception as e:
            logger.error(f"Failed to load sector mappings from {path}: {e}")
            return ({}, {}, [])
