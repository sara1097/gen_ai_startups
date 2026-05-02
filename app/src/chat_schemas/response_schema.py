from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ==========================================================
# INTENT
# ==========================================================

class IntentDetail(BaseModel):
    intent: str
    confidence: str
    relevant_text: str
    priority: int


class IntentSchema(BaseModel):
    detected_intents: List[IntentDetail]
    primary_intent: str = Field(..., description="Main detected intent")
    secondary_intents: List[str] = Field(default_factory=list)


class ExtractedRequirements(BaseModel):
    core_problem: str
    requirements: List[str] = []
    references_previous: bool = False
    questions: List[str] = []
    constraints: List[str] = []

class IntentAndExtractionSchema(BaseModel):
    intent: IntentSchema
    extracted: ExtractedRequirements
    
# ==========================================================
# IDEA SCHEMA 
# ==========================================================

class BusinessModelSchema(BaseModel):
    value_proposition: str
    revenue_streams: List[str]
    pricing_model: str
    customer_acquisition: List[str]


class MarketAnalysisSchema(BaseModel):
    market_size: str
    competitors: List[str]
    competitive_advantage: str


class FeasibilitySchema(BaseModel):
    technical_feasibility: str
    market_feasibility: str
    risk_factors: List[str]


class ImpactSchema(BaseModel):
    economic_impact: str
    social_impact: str


class MvpPlanSchema(BaseModel):
    mvp_features: List[str]
    first_steps: List[str]


class IdeaSchema(BaseModel):
    problem_title: str
    problem_description: str
    root_cause: str
    target_users: str
    market_region: str = Field(default="Egypt or MENA")
    why_now: str

    solution_name: str
    solution_description: str
    key_features: List[str]
    technology_stack: List[str]

    business_model: BusinessModelSchema
    market_analysis: MarketAnalysisSchema
    feasibility: FeasibilitySchema

    novelty_score: int = Field(ge=0, le=100)

    impact: ImpactSchema
    mvp_plan: MvpPlanSchema


# ==========================================================
# 🔹 REQUEST
# ==========================================================

class ChatRequest(BaseModel):
    content: str = Field(
        ..., 
        description="User message",
        min_length=1,
        examples=[
            "I want to solve expensive education in Egypt",
            "Give me a startup idea",
            "Tell me more about the business model"
        ]
    )

    conversationId: str = Field(
        ...,
        description="Conversation ID",
        examples=["123", "456"]
    )

    clientMessageId: Optional[str] = Field(
        default=None,
        description="Client message ID"
    )

    isNewConversation: bool = Field(
        ...,
        description="Is new conversation"
    )

    domain: Optional[str] = Field(
        default=None,
        description="Business domain"
    )

    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Previous idea data"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "content": "I want to solve expensive education in Egypt",
                    "conversationId": "123",
                    "clientMessageId": "msg_001",
                    "isNewConversation": True,
                    "domain": "education",
                    "data": None
                },
                {
                    "content": "Give me a startup idea in healthcare",
                    "conversationId": "124",
                    "clientMessageId": "msg_002",
                    "isNewConversation": True,
                    "domain": "healthcare",
                    "data": None
                }
            ]
        }


# ==========================================================
#  RESPONSE
# ==========================================================

class ChatResponse(BaseModel):
    content: str
    conversationId: str

    clientMessageId: Optional[str] = None
    conversation_title: Optional[str] = None

    role: str = "ai"
    is_idea_saved: bool = False
    is_full_idea: bool

    data: Optional[Dict[str, Any]] = None
    inspired_by: Optional[List[str]] = None

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "content": "EduFlex is an online education platform...",
                    "conversationId": "123",
                    "clientMessageId": "msg_001",
                    "conversation_title": "Education Solution",
                    "role": "ai",
                    "is_idea_saved": False,
                    "is_full_idea": True,
                    "data": {
                        "problem_title": "Expensive Education in Egypt",
                        "solution_name": "EduFlex",
                        "problem_description": "Students struggle with expensive education...",
                        "market_region": "Egypt/MENA"
                    },
                    "inspired_by": ["affordable learning", "online education"]
                },
                {
                    "content": "The business model includes subscription fees...",
                    "conversationId": "123",
                    "clientMessageId": "msg_003",
                    "conversation_title": None,
                    "role": "ai",
                    "is_idea_saved": False,
                    "is_full_idea": False,
                    "data": None,
                    "inspired_by": None
                }
            ]
        }