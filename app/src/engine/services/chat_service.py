import logging
import pandas as pd

from app.src.chat_schemas.response_schema import ChatRequest, ChatResponse, IdeaSchema
from app.src.engine.core.parsers import QwenParser
from app.src.llm.groq_provider import GroqProvider
from app.src.engine.rag.retriever import StartupRetriever
from app.src.engine.core.intent_classification import IntentClassifier
from app.src.prompt_Engineering.templates import PromptBuilder, build_idea_user_prompt, IDEA_GENERATION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ChatService:

    def __init__(self, llm_provider: GroqProvider, retriever: StartupRetriever, intent_classifier: IntentClassifier):
        self.llm = llm_provider
        self.retriever = retriever
        self.classifier = intent_classifier
        self.prompt_builder = PromptBuilder()

    def process_message(self, request: ChatRequest) -> ChatResponse:

        logger.info(f"Processing message {request.conversationId}")

        # =========================
        # 1. INTENT + EXTRACTION
        # =========================
        # intent_data = self.classifier.classify(request.content)
        # extracted = self.classifier.extract_requirements(request.content)

        combined = self.classifier.classify_and_extract(request.content)

        intent_data = combined.intent
        extracted = combined.extracted
        
        primary_intent = intent_data.primary_intent

        # =========================
        # 2. LANGUAGE DETECTION 
        # =========================
        lang = self._detect_language(request.content)

        # =========================
        # 3. RANDOM PROBLEM 
        # =========================

        user_problem = extracted.core_problem

        no_problem = (
            not user_problem or
            user_problem.strip().lower() == request.content.strip().lower()
            )

        # if primary_intent == "random_solution":
        if primary_intent == "random_solution" or no_problem:

            try:
                df = pd.read_excel("data/raw/Problems.xlsx")

                filtered = df[
                    df['problem_sector'].str.lower().isin(
                        [d.lower() for d in request.domain] if request.domain else []
                    )
                ]

                if not filtered.empty:
                    random_problem = filtered.sample(n=1)['problem_description'].values[0]
                    extracted = self.classifier.extract_requirements(random_problem)
                    logger.info(f"Random problem selected: {random_problem}")

            except Exception as e:
                logger.warning(f"Random problem failed: {e}")

        # =========================
        # 4. RAG CONTEXT
        # =========================
        context_points = []
        context_text = ""


        
    
        # if primary_intent in ["problem_solving", "random_solution", "alternative_idea"]:
        if not no_problem and primary_intent in ["problem_solving", "alternative_idea"]:
            context_points = self.retriever.retrieve_topk(
                problem_text=extracted.core_problem or request.content,
                sector=request.domain
            )

            # context_text = self._make_context_cards(context_points)
        

        # =========================
        # GUARD: GENERAL CHAT / GREETING
        #   =========================
        user_text = request.content.lower().strip()
        if primary_intent == "general_chat" and len(user_text.split()) <= 5:

            messages = [
                {"role": "system", "content": "You are a friendly assistant. Keep responses short and natural."},
                {"role": "user", "content": request.content}
                 ]
            response = self.llm.generate(messages, temperature=0.7)

            return ChatResponse(
                content=response,
                conversationId=request.conversationId,
                conversation_title=None,
                role='ai',
                is_idea_saved=False,
                is_full_idea=False,
                data=None,
                inspired_by=None
            )
        # if primary_intent == "general_chat" and len(user_text.split()) <= 5:
        #     return ChatResponse(
        #         content="Hey 👋 How can I help you with a startup idea?",
        #         conversationId=request.conversationId,
        #         conversation_title=None,
        #         role='ai',
        #         is_idea_saved=False,
        #         is_full_idea=False,
        #         data=None,
        #         inspired_by=None
        #     )

        # =========================
        # 5. IDEA GENERATION
        # =========================
        idea_data = None
        is_idea = primary_intent in ["problem_solving", "random_solution", "alternative_idea"]

        if is_idea:
            idea_data = self._generate_idea_data(extracted.core_problem or request.content)

        elif request.data:
            try:
                idea_data = IdeaSchema(**request.data)
            except Exception:
                idea_data = request.data

        # =========================
        # 6. FINAL PROMPT
        # =========================
        final_prompt = self.prompt_builder.build_unified_prompt(
            intent_data=intent_data,
            extracted_data=extracted,
            context_points=context_points,   
            idea_data=idea_data.model_dump() if isinstance(idea_data, IdeaSchema) else idea_data
        )

        messages = [
            {"role": "system", "content": self.prompt_builder.RESPONSE_SYSTEM_PROMPT},
            {"role": "user", "content": final_prompt}
        ]

        # content = self.llm.generate(messages, temperature=0.5)
        raw_response=self.llm.generate(messages, temperature=0.6)
        content = QwenParser.remove_think_blocks(raw_response)
        # =========================
        # 7. RESPONSE
        # =========================
        conversation_title = extracted.core_problem if request.isNewConversation else None

        inspired_by = [p.get("name", "") for p in context_points] if context_points else None

        return ChatResponse(
            content=content,
            conversationId=request.conversationId,
            conversation_title=conversation_title,
            role='ai',
            is_idea_saved=False,
            is_full_idea=is_idea,
            data=idea_data.model_dump() if isinstance(idea_data, IdeaSchema) else idea_data,
            inspired_by=inspired_by
        )

    # =========================
    # HELPERS 
    # =========================

    def _make_context_cards(self, points):
        if not points:
            return ""

        cards = []
        for i, p in enumerate(points, 1):
            card = f"""[{i}]
name: {p.get("name","")}
domain: {p.get("domain","")}
use_case: {p.get("use_case","")}
solution: {p.get("solution","")}
"""
            cards.append(card.strip())

        return "\n\n".join(cards)

    def _detect_language(self, text: str) -> str:
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return "Arabic" if arabic_chars > len(text) * 0.2 else "English"

    def _generate_idea_data(self, problem: str) -> IdeaSchema:
        messages = [
            {"role": "system", "content": IDEA_GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": build_idea_user_prompt(problem)}
        ]

        raw = self.llm.generate_structured(messages, temperature=0.2)
        return QwenParser.parse_and_validate(raw, IdeaSchema)