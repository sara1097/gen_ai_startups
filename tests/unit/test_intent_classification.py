import pytest
from unittest.mock import patch, MagicMock
import json


class TestClassifyIntent:

    @patch("app.src.engine.core.intent_classification.llm_provider")
    def test_returns_primary_intent(self, mock_llm):
        """classify_intent لازم يرجع primary_intent"""
        mock_llm.generate.return_value = json.dumps({
            "detected_intents": [{"intent": "problem_solving", "confidence": "high", "relevant_text": "test", "priority": 1}],
            "primary_intent": "problem_solving",
            "secondary_intents": []
        })

        from app.src.engine.core.intent_classification import classify_intent
        result = classify_intent("I need help with payment processing")

        assert "primary_intent" in result
        assert result["primary_intent"] == "problem_solving"

    @patch("app.src.engine.core.intent_classification.llm_provider")
    def test_fallback_on_bad_json(self, mock_llm):
        """لو الـ LLM رجع JSON غلط يرجع default intent"""
        mock_llm.generate.return_value = "this is not json at all"

        from app.src.engine.core.intent_classification import classify_intent
        result = classify_intent("hello")

        # لازم يرجع الـ default بدل ما يكسر
        assert result["primary_intent"] == "general_chat"

    @patch("app.src.engine.core.intent_classification.llm_provider")
    def test_extract_problem_returns_core_problem(self, mock_llm):
        """extract_problem_and_requirements لازم يرجع core_problem"""
        mock_llm.generate.return_value = json.dumps({
            "core_problem": "digital payment inefficiency",
            "requirements": ["fast", "secure"],
            "references_previous": False,
            "questions": [],
            "constraints": []
        })

        from app.src.engine.core.intent_classification import extract_problem_and_requirements
        result = extract_problem_and_requirements("payment is slow and insecure")

        assert result["core_problem"] == "digital payment inefficiency"
        assert isinstance(result["requirements"], list)

    @patch("app.src.engine.core.intent_classification.llm_provider")
    def test_extract_problem_fallback_on_error(self, mock_llm):
        """لو الـ LLM فشل يرجع default extraction"""
        mock_llm.generate.side_effect = Exception("LLM Error")

        from app.src.engine.core.intent_classification import extract_problem_and_requirements
        result = extract_problem_and_requirements("some input")

        # لازم يرجع dict مش يكسر
        assert isinstance(result, dict)
        assert "core_problem" in result