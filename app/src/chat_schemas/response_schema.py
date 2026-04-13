from pydantic import BaseModel
from typing import Optional, List, Dict

class IntentSchema(BaseModel):
    primary_intent: str
    secondary_intents: List[str] = []

class ChatResponse(BaseModel):
    content: str
    conversationId: str
    conversation_title:Optional[str]
    role: str = 'ai',
    is_idea_saved: bool = False
    is_full_idea: bool
    data: Optional[Dict] = None
    inspired_by: Optional[List[str]] = None


class ChatRequest(BaseModel):
    content: str
    conversationId: str
    isNewConversation: bool 
    clientMessageId: str = None
    domain: Optional[str] = None
    data: Optional[Dict] = None


