import json
import re
import logging
from typing import Dict, Optional

from app.src.llm.groq_provider import groq_provider
from app.src.prompt_Engineering.tamplates import INTENTS_DETECTION_TEMPLATE

logger = logging.getLogger(__name__)
llm_provider = groq_provider()

# ─── Constants ────────────────────────────────────────────────────────────────

VALID_INTENTS = {
    "problem_solving", "random_solution",
    "follow_up", "alternative_idea", "general_chat"
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
        r"\b(hi|hello|hey|مرحبا|السلام|كيف|what can you|ايه اللي|بتعمل)\b"
    ]
}

# ─── Public API ────────────────────────────────────────────────────────────────

def classify_intent(message: str, idea_data: Optional[Dict] = None) -> Dict:
    """
    Classify user intent using LLM + 4-layer guard system.
    Always returns a safe, validated intent dict.
    """
    logger.info("Classifying intent")

    raw = _call_llm(message)
    guarded = _run_guards(raw, message, idea_data)

    if "_guard_note" in guarded:
        logger.warning(f"[IntentGuard] {guarded['_guard_note']}")

    logger.info(f"Final intent: {guarded['primary_intent']}")
    return guarded


# ─── LLM Call ─────────────────────────────────────────────────────────────────

def _call_llm(message: str) -> Dict:
    try:
        response = llm_provider.generate([
            {"role": "user", "content": INTENTS_DETECTION_TEMPLATE.format(user_message=message)}
        ])
        logger.debug(f"Raw LLM response: {response}")
        cleaned = _clean_json_response(response)
        return json.loads(cleaned)

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        return _fallback(message, reason="json_decode_error")

    except Exception as e:
        logger.exception(f"LLM call failed: {type(e).__name__}: {e}")
        return _fallback(message, reason="llm_exception")


# ─── Guard Pipeline ────────────────────────────────────────────────────────────

def _run_guards(result: Dict, user_input: str, idea_data: Optional[Dict]) -> Dict:
    result = _guard_schema(result, user_input)
    result = _guard_confidence(result)
    result = _guard_rules(result, user_input)
    result = _guard_context(result, idea_data)
    return result


def _guard_schema(result: Dict, user_input: str) -> Dict:
    """Layer 1 — validates JSON structure and intent values."""
    try:
        primary = result.get("primary_intent", "")
        intents = result.get("detected_intents", [])

        if primary not in VALID_INTENTS:
            return _fallback(user_input, reason=f"invalid primary_intent: '{primary}'")

        if not isinstance(intents, list) or not intents:
            return _fallback(user_input, reason="empty detected_intents")

        for item in intents:
            if not isinstance(item, dict):
                return _fallback(user_input, reason="malformed intent item")
            if item.get("intent") not in VALID_INTENTS:
                return _fallback(user_input, reason=f"unknown intent: {item.get('intent')}")
            if item.get("confidence") not in VALID_CONFIDENCE:
                item["confidence"] = "low"  # repair بدل فشل كامل

        return result

    except Exception as e:
        return _fallback(user_input, reason=f"schema_guard_exception: {e}")


def _guard_confidence(result: Dict) -> Dict:
    """Layer 2 — low-confidence primary intent → demote to general_chat."""
    primary = result.get("primary_intent")
    intents = result.get("detected_intents", [])

    primary_obj = next((i for i in intents if i["intent"] == primary), None)

    if primary_obj and primary_obj.get("confidence") == "low":
        result["primary_intent"] = "general_chat"
        result["_guard_note"] = f"confidence_guard: '{primary}' demoted (low confidence)"

    return result


def _guard_rules(result: Dict, user_input: str) -> Dict:
    """Layer 3 — keyword cross-check when LLM isn't confident."""
    primary = result.get("primary_intent")
    intents = result.get("detected_intents", [])

    primary_obj = next((i for i in intents if i["intent"] == primary), None)
    confidence = primary_obj.get("confidence") if primary_obj else "low"

    if confidence == "high":
        return result  # LLM واثق → لا تدخل

    for intent, patterns in RULE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                if intent != primary:
                    result["_guard_note"] = f"rule_guard: '{primary}' → '{intent}'"
                    result["primary_intent"] = intent
                return result

    return result


def _guard_context(result: Dict, idea_data: Optional[Dict]) -> Dict:
    """Layer 4 — follow_up/alternative_idea need existing idea_data."""
    primary = result.get("primary_intent")

    if primary in ("follow_up", "alternative_idea") and not idea_data:
        result["_guard_note"] = f"context_guard: '{primary}' → 'random_solution' (no idea_data)"
        result["primary_intent"] = "random_solution"

    return result


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _fallback(user_input: str, reason: str = "") -> Dict:
    return {
        "detected_intents": [{
            "intent": "general_chat",
            "confidence": "high",
            "relevant_text": user_input,
            "priority": 1
        }],
        "primary_intent": "general_chat",
        "secondary_intents": [],
        "_guard_note": f"fallback: {reason}"
    }


def _clean_json_response(response: str) -> str:
    response = re.sub(r'```(?:json|python|text)?\s*\n?', '', response)
    response = re.sub(r'\n?```', '', response)
    start = response.find('{')
    if start != -1:
        response = response[start:]
    end = response.rfind('}')
    if end != -1:
        response = response[:end + 1]
    return response.strip()


# ─── Unchanged helpers (kept as-is) ───────────────────────────────────────────

def extract_problem_and_requirements(user_input: str) -> Dict:
    logger.info("Extracting problem and requirements")
    extraction_prompt = f"""You are a multilingual assistant. The user may write in Arabic or English.

User input: "{user_input}"

Return ONLY valid JSON with no other text:
{{
  "core_problem": "",
  "requirements": []
}}

Instructions:
- Always write all extracted text in English
- "core_problem": Full descriptive sentence preserving ALL context from user input
- Never reduce the problem to a single word
- Preserve domain-specific details (clinic, school, transport, etc.)
"""
    try:
        response = llm_provider.generate([
            {"role": "user", "content": extraction_prompt}
        ])
        logger.debug(f"Raw extraction response: {response}")
        cleaned = _extract_json_only(response)
        parsed = json.loads(cleaned)
        logger.info("Extraction succeeded")
        return {
            "core_problem": parsed.get("core_problem", ""),
            "requirements": parsed.get("requirements", [])
        }
    except json.JSONDecodeError as e:
        logger.error(f"JSON Parse Error: {e}")
        return _get_default_extraction(user_input)
    except Exception as e:
        logger.exception(f"Extraction error: {type(e).__name__}: {e}")
        return _get_default_extraction(user_input)


def _extract_json_only(text: str) -> str:
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    start = text.find('{')
    if start == -1:
        return '{}'
    count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            count += 1
        elif text[i] == '}':
            count -= 1
            if count == 0:
                return text[start:i + 1]
    return '{}'


def _get_default_extraction(user_input: str) -> Dict:
    return {
        "core_problem": "",
        "requirements": [],
        "references_previous": False,
        "questions": [user_input],
        "constraints": []
    }


def detect_language(text: str) -> str:
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    return "Arabic" if arabic_chars > len(text) * 0.2 else "English"