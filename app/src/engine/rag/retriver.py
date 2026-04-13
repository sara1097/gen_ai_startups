
import os

from qdrant_client import QdrantClient, models
from qdrant_client.models import Prefetch, FusionQuery, Fusion
from deep_translator import GoogleTranslator
import langdetect

import yaml
import logging

from app.src.engine.core.providers.providers_factory import ProviderFactory

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv(".env")

def load_sector_mappings():
    path = "app/config/domain_mapping.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return (
        data["STARTUP_SECTOR_GROUPS"],
        data["PROBLEM_TO_STARTUP_GROUPS"],
        data["BOILERPLATE_SIGNALS"]
    )

def load_models_names():
    path = "app/config/model_config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return (
        data["encoder_model"],
        data["reranker"],
        data["sparse_model"]
    )

STARTUP_SECTOR_GROUPS, PROBLEM_TO_STARTUP_GROUPS, BOILERPLATE_SIGNALS  = load_sector_mappings()

def get_startup_sectors_for_problem(problem_sector: str) -> list[str]:
    logger.info(f"problem Sector mapping")
    group_names = PROBLEM_TO_STARTUP_GROUPS.get(problem_sector, [])
    sectors = []
    for g in group_names:
        sectors.extend(STARTUP_SECTOR_GROUPS.get(g, []))
    return list(set(sectors))

encoder_model_name, reranker_name, sparse_model_name = load_models_names()

providers = ProviderFactory()

embedding_provider = providers.embedding
reranker_provider = providers.reranker
sparse_provider = providers.sparse


def is_boilerplate(payload: dict) -> bool:
    text = " ".join([payload.get("use_case",""), payload.get("solution",""), payload.get("description","")]).lower()
    return any(s in text for s in BOILERPLATE_SIGNALS)

def translate_to_english(text: str) -> str:
    try:
        if langdetect.detect(text) == "ar":
            translated = GoogleTranslator(source="ar", target="en").translate(text)
            logger.debug(f"Translated: {translated}")
            return translated
    except Exception:
        pass
    return text

qdrant_client= QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

def retrieve_topk(
    problem_text: str,
    k: int = 5,
    sector: str | None = None,
    topN: int = 150,
    debug: bool = True
):
    logger.info(f"Getting The top 5 Startups")
    problem_en = translate_to_english(problem_text)
    ce_query   = f"{sector}: {problem_en}" if sector else problem_en

    dense_vec  = embedding_provider.encode(problem_en)
    sparse_vec = sparse_provider.encode(problem_en)

    # Soft sector filter (SHOULD = boost, not hard exclusion)
    startup_sectors = get_startup_sectors_for_problem(sector) if sector else []
    soft_filter = None
    if startup_sectors:
        soft_filter = models.Filter(
            should=[models.FieldCondition(
                key="sector",
                match=models.MatchAny(any=startup_sectors[:50])
            )]
        )
        if debug:
            logger.debug(f"'{sector}' → {len(startup_sectors)} startup sectors boosted")
    else:
        if debug:
            logger.warning(f"No mapping for '{sector}' — searching without sector boost")

    def run_query(use_filter):
        return qdrant_client.query_points(
            collection_name= os.getenv("COLLECTION"),
            prefetch=[
                Prefetch(query=dense_vec,  using="dense",  limit=topN, filter=use_filter),
                Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist()
                    ),
                    using="sparse", limit=topN, filter=use_filter
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=topN,
            with_payload=True,
        )

    results = run_query(soft_filter)

    # Fallback: if fewer than k results, retry without filter
    if len(results.points) < k and soft_filter is not None:
        if debug:
            logger.warning(f"Only {len(results.points)} results with filter — retrying without")
        results = run_query(None)

    # Clean: remove boilerplate + dedup by name
    seen, clean, skipped = set(), [], 0
    for p in results.points:
        # if is_boilerplate(p.payload):
        #     skipped += 1
        #     continue
        name = (p.payload.get("name") or "").strip().lower()
        if name not in seen:
            seen.add(name)
            clean.append(p)

    if debug:
        logger.debug(f"{len(results.points)} retrieved → {skipped} boilerplate removed → {len(clean)} unique clean")
        logger.debug(f"{len(results.points)}  → {len(clean)} unique clean")

    # Cross-encoder rerank
    pairs = [[ce_query, " | ".join(filter(bool, [
        p.payload.get("use_case",""),
        p.payload.get("solution",""),
        p.payload.get("description",""),
        p.payload.get("sector",""),
    ]))] for p in clean]

    cross_scores = [
        reranker_provider.score(q, d)
        for q, d in pairs
    ]
    ranked = sorted(zip(cross_scores, clean), key=lambda x: x[0], reverse=True)

    if debug:
        logger.debug(f"\n=== TOP-{k} ===")
        for score, p in ranked[:k]:
            pl = p.payload
            logger.debug(f"  {round(float(score),3):>7} | {pl.get('name',''):<28} | {pl.get('sector',''):<22} | {pl.get('domain','')}")
            logger.debug(f"           {str(pl.get('use_case',''))[:110]}")

    return [p for _, p in ranked[:k]]