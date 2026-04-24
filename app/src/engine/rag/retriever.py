# app/src/engine/rag/retriever.py

import logging
from typing import List, Optional, Any
from qdrant_client import QdrantClient, models
from qdrant_client.models import Prefetch, FusionQuery, Fusion
from deep_translator import GoogleTranslator
import langdetect

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class StartupRetriever:
    """
    Advanced RAG Retriever:
    - Hybrid search (dense + sparse)
    - Sector boosting
    - Query expansion
    - Deduplication
    - Optional boilerplate filtering (commented)
    - Cross-encoder reranking
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_provider: Any,
        sparse_provider: Any,
        reranker_provider: Any,
        sector_mappings: tuple
    ):
        self.settings = get_settings()
        self.client = qdrant_client
        self.embedding = embedding_provider
        self.sparse = sparse_provider
        self.reranker = reranker_provider

        self.startup_sector_groups, self.problem_to_startup_groups, self.boilerplate_signals = sector_mappings

    # =========================
    # SECTOR MAPPING
    # =========================
    def get_startup_sectors_for_problem(self, problem_sector: str) -> List[str]:
        if not problem_sector:
            return []

        logger.info(f"Mapping problem sector: {problem_sector}")
        group_names = self.problem_to_startup_groups.get(problem_sector, [])

        sectors = []
        for g in group_names:
            sectors.extend(self.startup_sector_groups.get(g, []))

        return list(set(sectors))

    # =========================
    # TRANSLATION
    # =========================
    def translate_to_english(self, text: str) -> str:
        if not text:
            return text

        try:
            if langdetect.detect(text) == "ar":
                translated = GoogleTranslator(source="ar", target="en").translate(text)
                logger.debug(f"Translated query: {translated}")
                return translated
        except Exception as e:
            logger.warning(f"Translation failed: {e}")

        return text

    # =========================
    # BOILERPLATE FILTER (COMMENTED AS REQUESTED)
    # =========================
    def is_boilerplate(self, payload: dict) -> bool:
        text = " ".join([
            payload.get("use_case", ""),
            payload.get("solution", ""),
            payload.get("description", "")
        ]).lower()

        return any(signal in text for signal in self.boilerplate_signals)

    # =========================
    # MAIN RETRIEVAL
    # =========================
    def retrieve_topk(
        self,
        problem_text: str,
        k: int = 5,
        sector: Optional[str] = None,
        topN: int = 150,
        debug: bool = True
    ) -> List[Any]:

        logger.info(f"Retrieving top {k} startups for: {problem_text[:50]}")

        #  1. Translation + Query Expansion
        problem_en = self.translate_to_english(problem_text)
        expanded_query = f"{problem_en}. startup solution, innovation, business idea"

        ce_query = f"{sector}: {expanded_query}" if sector else expanded_query

        #  2. Embeddings
        dense_vec = self.embedding.encode(problem_en)
        sparse_vec = self.sparse.encode(problem_en)

        #  3. Sector Boost
        startup_sectors = self.get_startup_sectors_for_problem(sector)
        soft_filter = None

        if startup_sectors:
            soft_filter = models.Filter(
                should=[models.FieldCondition(
                    key="sector",
                    match=models.MatchAny(any=startup_sectors[:50])
                )]
            )
            if debug:
                logger.debug(f"Boosting {len(startup_sectors)} sectors for '{sector}'")
        else:
            if debug:
                logger.warning(f"No mapping for '{sector}'")

        #  4. Query Function
        def run_query(use_filter):
            return self.client.query_points(
                collection_name=self.settings.COLLECTION_NAME,
                prefetch=[
                    Prefetch(query=dense_vec, using="dense", limit=topN, filter=use_filter),
                    Prefetch(
                        query=models.SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist()
                        ),
                        using="sparse",
                        limit=topN,
                        filter=use_filter
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=topN,
                with_payload=True,
            )

        try:
            #  5. Execute
            results = run_query(soft_filter)

            #  6. Fallback
            if len(results.points) < k and soft_filter is not None:
                logger.warning("Few results → retry without filter")
                results = run_query(None)

            # 7. Clean + Deduplicate
            seen, clean, skipped = set(), [], 0

            for p in results.points:
                if p.payload is None:
                    skipped += 1
                    continue

                
                # if self.is_boilerplate(p.payload):
                #     skipped += 1
                #     continue

                name = (p.payload.get("name") or "").strip().lower()

                if name and name not in seen:
                    seen.add(name)
                    clean.append(p)

            if debug:
                logger.debug(f"{len(results.points)} → {len(clean)} clean (skipped {skipped})")

            if not clean:
                return []

            #  8. Reranking
            pairs = [[ce_query, " | ".join(filter(bool, [
                p.payload.get("use_case", ""),
                p.payload.get("solution", ""),
                p.payload.get("description", ""),
                p.payload.get("sector", ""),
            ]))] for p in clean]

            scores = [self.reranker.score(q, d) for q, d in pairs]
            ranked = sorted(zip(scores, clean), key=lambda x: x[0], reverse=True)

            #  9. Debug TOP-K
            if debug:
                logger.debug("\n=== TOP RESULTS ===")
                for score, p in ranked[:k]:
                    pl = p.payload
                    logger.debug(
                        f"{round(float(score),3):>6} | {pl.get('name',''):<25} | {pl.get('sector',''):<20}"
                    )

            # 10. Context Cards 
            final_results = []

            for score, p in ranked[:k]:
                payload = p.payload

                payload["context_card"] = f"""
name: {payload.get("name")}
domain: {payload.get("domain")}
use_case: {payload.get("use_case")}
solution: {payload.get("solution")}
score: {round(float(score), 3)}
""".strip()

                final_results.append(payload)

            return final_results

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []