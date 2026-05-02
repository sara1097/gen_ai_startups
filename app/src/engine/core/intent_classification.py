# app/src/engine/core/intent_classification.py

import logging
import re
from typing import Optional

from app.src.chat_schemas.response_schema import IntentAndExtractionSchema, IntentSchema, ExtractedRequirements
from app.src.engine.core.parsers import QwenParser
from app.src.llm.groq_provider import GroqProvider
from app.src.prompt_Engineering.templates import (
    INTENT_SYSTEM_PROMPT,
    build_intent_user_prompt,
    PROBLEM_EXTRACTION_SYSTEM_PROMPT,
    build_problem_extraction_prompt,
    INTENT_AND_EXTRACTION_PROMPT
)

logger = logging.getLogger(__name__)

# ===== Guards Config =====
VALID_INTENTS = {
    "problem_solving", "random_solution",
    "follow_up", "alternative_idea", "details",
    "feasibility", "novelty", "general_chat"
}

VALID_CONFIDENCE = {"high", "medium", "low"}


RULE_PATTERNS = {
    "problem_solving": [
        r"\b(solve|fix|problem|issue|عندي|مشكلة|حل|عايز أحل)\b"
    ],
    "random_solution": [
        r"\b(give me|startup idea|business idea|فكرة|مشروع|ابدأ)\b"
    ],
    "follow_up": [
        r"\b(more|tell me|expand|explain|أكثر|وضح|تفاصيل|كمّل)\b"
    ],
    "general_chat": [
        r"\b(hi|hello|hey|مرحبا|السلام|كيف|what can you|ايه اللي|بتعمل)\b",

        r"\b(can i tell you|can i share|is it okay|هقدر|ممكن أقولك|عايز أقولك)\b",
    ]
}



class IntentClassifier:
    def __init__(self, llm_provider: GroqProvider):
        self.llm = llm_provider

    # =========================
    # MAIN CLASSIFY FUNCTION
    # =========================
    def classify(self, user_input: str, idea_data: Optional[dict] = None) -> IntentSchema:
        logger.info("Classifying intent...")

        messages = [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": build_intent_user_prompt(user_input)}
        ]

        try:
            raw_response = self.llm.generate_structured(messages, temperature=0.1)
            parsed = QwenParser.parse_and_validate(raw_response, IntentSchema)

            #  Apply Guards
            parsed = self._apply_guards(parsed, user_input, idea_data)

            return parsed

        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return self._get_default_intent(user_input)

    # =========================
    # REQUIREMENTS EXTRACTION
    # =========================
    def extract_requirements(self, user_input: str) -> ExtractedRequirements:
        logger.info("Extracting requirements...")

        messages = [
            {"role": "system", "content": PROBLEM_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": build_problem_extraction_prompt(user_input)}
        ]

        try:
            raw_response = self.llm.generate_structured(messages, temperature=0.1)
            return QwenParser.parse_and_validate(raw_response, ExtractedRequirements)

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return self._get_default_extraction(user_input)


    def classify_and_extract(self, user_input: str):
        messages = [
            {"role": "system", "content": INTENT_AND_EXTRACTION_PROMPT},
            {"role": "user", "content": user_input}
        ]

        raw = self.llm.generate_structured(messages, temperature=0.1)

        parsed = QwenParser.parse_and_validate(
            raw,
            IntentAndExtractionSchema
        )

        return parsed
    # =========================
    # GUARDS SYSTEM 
    # =========================
    def _apply_guards(self, result: IntentSchema, user_input: str, idea_data: Optional[dict]) -> IntentSchema:
        result = self._guard_schema(result, user_input)
        result = self._guard_confidence(result)
        result = self._guard_rules(result, user_input)
        result = self._guard_context(result, idea_data)
        return result

    def _guard_schema(self, result: IntentSchema, user_input: str) -> IntentSchema:
        try:
            if result.primary_intent not in VALID_INTENTS:
                return self._get_default_intent(user_input)

            for item in result.detected_intents:
                if item.intent not in VALID_INTENTS:
                    return self._get_default_intent(user_input)

                if item.confidence not in VALID_CONFIDENCE:
                    item.confidence = "low"

            return result

        except Exception:
            return self._get_default_intent(user_input)

    def _guard_confidence(self, result: IntentSchema) -> IntentSchema:
        primary = result.primary_intent
        primary_obj = next((i for i in result.detected_intents if i.intent == primary), None)

        if primary_obj and primary_obj.confidence == "low":
            result.primary_intent = "general_chat"

        return result

    def _guard_rules(self, result: IntentSchema, user_input: str) -> IntentSchema:
        primary = result.primary_intent
        primary_obj = next((i for i in result.detected_intents if i.intent == primary), None)

        if primary_obj and primary_obj.confidence == "high":
            return result

        for intent, patterns in RULE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, user_input, re.IGNORECASE):
                    result.primary_intent = intent
                    return result

        return result

    def _guard_context(self, result: IntentSchema, idea_data: Optional[dict]) -> IntentSchema:
        if result.primary_intent in ["follow_up", "alternative_idea"] and not idea_data:
            result.primary_intent = "random_solution"

        return result

    # =========================
    # FALLBACKS
    # =========================
    def _get_default_intent(self, user_input: str) -> IntentSchema:
        return IntentSchema(
            detected_intents=[{
                "intent": "general_chat",
                "confidence": "high",
                "relevant_text": user_input,
                "priority": 1
            }],
            primary_intent="general_chat",
            secondary_intents=[]
        )

    def _get_default_extraction(self, user_input: str) -> ExtractedRequirements:
        return ExtractedRequirements(
            core_problem=user_input,
            requirements=[],
            references_previous=False,
            questions=[],
            constraints=[]
        )