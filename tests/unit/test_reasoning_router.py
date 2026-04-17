import pytest
import json
from unittest.mock import patch, MagicMock


# ============================================================
# Helper — بيعمل mock point جاهز نستخدمه في كل test
# ============================================================
def make_mock_point(name="TestStartup"):
    point = MagicMock()
    point.payload = {
        "name": name,
        "domain": "finance",
        "use_case": "payment processing",
        "solution": "AI payments",
        "link": "https://example.com"
    }
    return point


# ============================================================
# الـ fixtures المشتركة بين كل الـ tests
# ============================================================
@pytest.fixture
def mock_intent_problem_solving():
    """intent = problem_solving"""
    return {
        "detected_intents": [{"intent": "problem_solving", "confidence": "high", "relevant_text": "test", "priority": 1}],
        "primary_intent": "problem_solving",
        "secondary_intents": []
    }

@pytest.fixture
def mock_intent_general_chat():
    """intent = general_chat"""
    return {
        "detected_intents": [{"intent": "general_chat", "confidence": "high", "relevant_text": "hi", "priority": 1}],
        "primary_intent": "general_chat",
        "secondary_intents": []
    }

@pytest.fixture
def mock_intent_alternative():
    """intent = alternative_idea"""
    return {
        "detected_intents": [{"intent": "alternative_idea", "confidence": "high", "relevant_text": "alternative", "priority": 1}],
        "primary_intent": "alternative_idea",
        "secondary_intents": []
    }

@pytest.fixture
def mock_extracted():
    """الـ extracted problem الوهمي"""
    return {
        "core_problem": "digital payment inefficiency",
        "requirements": ["fast", "secure"],
        "references_previous": False,
        "questions": [],
        "constraints": []
    }

@pytest.fixture
def mock_idea_json():
    """الـ idea اللي LLM بيولدها"""
    return json.dumps({
        "startup_name": "PayFlow",
        "problem_description": "payment is slow",
        "solution": "AI-powered payment gateway",
        "target_market": "SMEs"
    })

@pytest.fixture
def base_request():
    """الـ request الأساسي المشترك"""
    return {
        "user_input": "I need a solution for digital payments",
        "data": {},
        "domain": "Finance",
        "isNewConversation": True,
        "conversationId": "conv-123"
    }


# ============================================================
# الـ Tests
# ============================================================
class TestRouteReasoning:

    # ----------------------------------------------------------
    # Test 1: problem_solving → يولد idea جديدة
    # ----------------------------------------------------------
    @patch("app.src.engine.core.reasoning_router.retrieve_topk")
    @patch("app.src.engine.core.reasoning_router.extract_problem_and_requirements")
    @patch("app.src.engine.core.reasoning_router.classify_intent")
    @patch("app.src.engine.core.reasoning_router.llm_provider")
    def test_problem_solving_generates_new_idea(
        self,
        mock_llm,
        mock_classify,
        mock_extract,
        mock_retrieve,
        mock_intent_problem_solving,
        mock_extracted,
        mock_idea_json,
        base_request
    ):
        """لو الـ intent = problem_solving لازم يولد idea جديدة"""
        # جهز الـ mocks
        mock_classify.return_value  = mock_intent_problem_solving
        mock_extract.return_value   = mock_extracted
        mock_retrieve.return_value  = [make_mock_point("Startup1"), make_mock_point("Startup2")]
        # الـ LLM بيتنادى مرتين: مرة للـ idea ومرة للـ content
        mock_llm.generate.side_effect = [mock_idea_json, "Here is your startup idea"]

        from app.src.engine.core.reasoning_router import route_reasoning
        result = route_reasoning(**base_request)

        # التأكدات
        assert result["content"] == "Here is your startup idea"
        assert result["is_full_idea"] == True
        assert result["role"] == "ai"
        # الـ LLM اتنادى مرتين
        assert mock_llm.generate.call_count == 2

    # ----------------------------------------------------------
    # Test 2: general_chat → ما يولدش idea
    # ----------------------------------------------------------
    @patch("app.src.engine.core.reasoning_router.retrieve_topk")
    @patch("app.src.engine.core.reasoning_router.extract_problem_and_requirements")
    @patch("app.src.engine.core.reasoning_router.classify_intent")
    @patch("app.src.engine.core.reasoning_router.llm_provider")
    def test_general_chat_does_not_generate_idea(
        self,
        mock_llm,
        mock_classify,
        mock_extract,
        mock_retrieve,
        mock_intent_general_chat,
        mock_extracted,
        base_request
    ):
        """لو الـ intent = general_chat ما يولدش startup idea"""
        mock_classify.return_value = mock_intent_general_chat
        mock_extract.return_value  = mock_extracted
        mock_retrieve.return_value = [make_mock_point()]
        # الـ LLM بيتنادى مرة واحدة بس للـ content
        mock_llm.generate.return_value = "Hello! How can I help?"

        base_request["data"] = {"existing": "data"}

        from app.src.engine.core.reasoning_router import route_reasoning
        result = route_reasoning(**base_request)

        assert result["is_full_idea"] == False
        assert mock_llm.generate.call_count == 1

    # ----------------------------------------------------------
    # Test 3: NewConversation → يرجع conversation_title
    # ----------------------------------------------------------
    @patch("app.src.engine.core.reasoning_router.retrieve_topk")
    @patch("app.src.engine.core.reasoning_router.extract_problem_and_requirements")
    @patch("app.src.engine.core.reasoning_router.classify_intent")
    @patch("app.src.engine.core.reasoning_router.llm_provider")
    def test_new_conversation_returns_title(
        self,
        mock_llm,
        mock_classify,
        mock_extract,
        mock_retrieve,
        mock_intent_problem_solving,
        mock_extracted,
        mock_idea_json,
        base_request
    ):
        """لو isNewConversation=True لازم يرجع conversation_title"""
        mock_classify.return_value = mock_intent_problem_solving
        mock_extract.return_value  = mock_extracted
        mock_retrieve.return_value = [make_mock_point()]
        mock_llm.generate.side_effect = [mock_idea_json, "response content"]

        base_request["isNewConversation"] = True

        from app.src.engine.core.reasoning_router import route_reasoning
        result = route_reasoning(**base_request)

        assert "conversation_title" in result
        assert result["conversation_title"] == "digital payment inefficiency"

    # ----------------------------------------------------------
    # Test 4: isNewConversation=False → ما يرجعش conversation_title
    # ----------------------------------------------------------
    @patch("app.src.engine.core.reasoning_router.retrieve_topk")
    @patch("app.src.engine.core.reasoning_router.extract_problem_and_requirements")
    @patch("app.src.engine.core.reasoning_router.classify_intent")
    @patch("app.src.engine.core.reasoning_router.llm_provider")
    def test_existing_conversation_no_title(
        self,
        mock_llm,
        mock_classify,
        mock_extract,
        mock_retrieve,
        mock_intent_general_chat,
        mock_extracted,
        base_request
    ):
        """لو isNewConversation=False ما يرجعش title"""
        mock_classify.return_value = mock_intent_general_chat
        mock_extract.return_value  = mock_extracted
        mock_retrieve.return_value = [make_mock_point()]
        mock_llm.generate.return_value = "response"

        base_request["isNewConversation"] = False

        from app.src.engine.core.reasoning_router import route_reasoning
        result = route_reasoning(**base_request)

        # conversation_title مش موجود أو None
        assert result.get("conversation_title") is None

    # ----------------------------------------------------------
    # Test 5: inspired_by بييجي من الـ retriever
    # ----------------------------------------------------------
    @patch("app.src.engine.core.reasoning_router.retrieve_topk")
    @patch("app.src.engine.core.reasoning_router.extract_problem_and_requirements")
    @patch("app.src.engine.core.reasoning_router.classify_intent")
    @patch("app.src.engine.core.reasoning_router.llm_provider")
    def test_inspired_by_comes_from_retriever(
        self,
        mock_llm,
        mock_classify,
        mock_extract,
        mock_retrieve,
        mock_intent_problem_solving,
        mock_extracted,
        mock_idea_json,
        base_request
    ):
        """الـ inspired_by لازم يكون أسماء الـ startups من الـ retriever"""
        mock_classify.return_value = mock_intent_problem_solving
        mock_extract.return_value  = mock_extracted
        mock_retrieve.return_value = [
            make_mock_point("StartupA"),
            make_mock_point("StartupB")
        ]
        mock_llm.generate.side_effect = [mock_idea_json, "content"]

        from app.src.engine.core.reasoning_router import route_reasoning
        result = route_reasoning(**base_request)

        assert result["inspired_by"] == ["StartupA", "StartupB"]

    # ----------------------------------------------------------
    # Test 6: alternative_idea → يستخدم problem من الـ data
    # ----------------------------------------------------------
    @patch("app.src.engine.core.reasoning_router.retrieve_topk")
    @patch("app.src.engine.core.reasoning_router.extract_problem_and_requirements")
    @patch("app.src.engine.core.reasoning_router.classify_intent")
    @patch("app.src.engine.core.reasoning_router.llm_provider")
    def test_alternative_idea_uses_existing_problem(
        self,
        mock_llm,
        mock_classify,
        mock_extract,
        mock_retrieve,
        mock_intent_alternative,
        mock_extracted,
        mock_idea_json,
        base_request
    ):
        """alternative_idea لازم ياخد الـ problem من الـ data مش من الـ input"""
        mock_classify.return_value = mock_intent_alternative
        mock_extract.return_value  = mock_extracted
        mock_retrieve.return_value = [make_mock_point()]
        mock_llm.generate.side_effect = [mock_idea_json, "alternative idea content"]

        # في الـ data فيه problem_description موجود
        base_request["data"] = {"problem_description": "existing problem from data"}

        from app.src.engine.core.reasoning_router import route_reasoning
        result = route_reasoning(**base_request)

        assert result["is_full_idea"] == True
        # الـ LLM اتنادى مرتين
        assert mock_llm.generate.call_count == 2

    # ----------------------------------------------------------
    # Test 7: لو الـ LLM رجع JSON غلط
    # ----------------------------------------------------------
    @patch("app.src.engine.core.reasoning_router.retrieve_topk")
    @patch("app.src.engine.core.reasoning_router.extract_problem_and_requirements")
    @patch("app.src.engine.core.reasoning_router.classify_intent")
    @patch("app.src.engine.core.reasoning_router.llm_provider")
    def test_handles_invalid_json_from_llm(
        self,
        mock_llm,
        mock_classify,
        mock_extract,
        mock_retrieve,
        mock_intent_problem_solving,
        mock_extracted,
        base_request
    ):
        """لو الـ LLM رجع JSON غلط ما يكسرش"""
        mock_classify.return_value = mock_intent_problem_solving
        mock_extract.return_value  = mock_extracted
        mock_retrieve.return_value = [make_mock_point()]
        # المرة الأولى: JSON غلط — المرة التانية: content عادي
        mock_llm.generate.side_effect = ["not valid json {{{{", "Here is the content"]

        from app.src.engine.core.reasoning_router import route_reasoning
        result = route_reasoning(**base_request)

        # ما كسرش وفيه raw_text في الـ data
        assert result is not None
        assert "raw_text" in result["data"]

    # ----------------------------------------------------------
    # Test 8: الـ response schema صح
    # ----------------------------------------------------------
    @patch("app.src.engine.core.reasoning_router.retrieve_topk")
    @patch("app.src.engine.core.reasoning_router.extract_problem_and_requirements")
    @patch("app.src.engine.core.reasoning_router.classify_intent")
    @patch("app.src.engine.core.reasoning_router.llm_provider")
    def test_response_has_required_fields(
        self,
        mock_llm,
        mock_classify,
        mock_extract,
        mock_retrieve,
        mock_intent_general_chat,
        mock_extracted,
        base_request
    ):
        """الـ response لازم يحتوي على كل الـ fields المطلوبة"""
        mock_classify.return_value = mock_intent_general_chat
        mock_extract.return_value  = mock_extracted
        mock_retrieve.return_value = [make_mock_point()]
        mock_llm.generate.return_value = "response"

        from app.src.engine.core.reasoning_router import route_reasoning
        result = route_reasoning(**base_request)

        required_fields = [
            "content",
            "conversationId",
            "role",
            "is_idea_saved",
            "is_full_idea",
            "data",
            "inspired_by"
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"