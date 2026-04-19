from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class IntentSchema(BaseModel):
    primary_intent: str = Field(..., description="The main intent detected")
    secondary_intents: List[str] = Field(
        default_factory=list,
        description="Additional intents detected"
    )


class ChatRequest(BaseModel):
    content: str = Field(
        ..., 
        description="The user's message or problem description",
        min_length=1,
        examples=[
            "I want to solve expensive education in Egypt",
            "Give me a startup idea",
            "Tell me more about the business model"
        ]
    )
    conversationId: int = Field(
        ...,
        description="Unique conversation identifier",
        examples=[123, 456, 789]
    )
    clientMessageId: str = Field(
        ...,
        description="Unique message identifier from client",
        examples=["msg_001", "msg_002"]
    )
    isNewConversation: bool = Field(
        default=False,
        description="Whether this starts a new conversation"
    )
    domain: list[str] = Field(
        default=["general"],
        description="Business domain/sector",
        examples=[["education", "healthcare"], ["technology", "transportation", "finance"]]
    )
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Previous idea data (for follow-up requests)"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "content": "I want to solve expensive education in Egypt",
                    "conversationId": 123,
                    "clientMessageId": "msg_001",
                    "isNewConversation": True,
                    "domain": "education",
                    "data": None
                },
                {
                    "content": "Give me a startup idea in healthcare",
                    "conversationId": 124,
                    "clientMessageId": "msg_002",
                    "isNewConversation": True,
                    "domain": "healthcare",
                    "data": None
                },
                {
                    "content": "Tell me more about the business model",
                    "conversationId": 123,
                    "clientMessageId": "msg_003",
                    "isNewConversation": False,
                    "domain": "education",
                    "data": {
                        "problem_title": "Expensive Education",
                        "solution_name": "EduFlex"
                    }
                }
            ]
        }


class ChatResponse(BaseModel):
    content: str = Field(
        ...,
        description="The AI response message"
    )
    conversationId: int = Field(
        ...,
        description="Conversation identifier"
    )
    clientMessageId: str = Field(
        ...,
        description="The message ID from the request"
    )
    conversation_title: Optional[str] = Field(
        default=None,
        description="Title for new conversations"
    )
    role: str = Field(
        default="ai",
        description="Role of the responder (ai/user)",
        examples=["ai"]
    )
    is_idea_saved: bool = Field(
        default=False,
        description="Whether the idea was saved to database"
    )
    is_full_idea: bool = Field(
        ...,
        description="Whether this response contains a full startup idea"
    )
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Full idea data structure (populated when is_full_idea=true)"
    )
    inspired_by: Optional[List[str]] = Field(
        default=None,
        description="List of problems or ideas that inspired this solution"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "content": "EduFlex is an online education platform...",
                    "conversationId": 123,
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
                    "conversationId": 123,
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