from app.src.engine.core.logger import setup_logging

setup_logging()

from dotenv import load_dotenv

from app.src.engine.core.reasoning_router import route_reasoning
load_dotenv(".env")

from app.src.chat_schemas.response_schema import ChatRequest, ChatResponse
from fastapi import FastAPI

from app.src.engine.core.reasoning_router import route_reasoning

app = FastAPI(title="Startup AI Service")

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    result = route_reasoning(
        user_input=request.content,
        data=request.data,
        isNewConversation=request.isNewConversation,
        clientMessageId = request.clientMessageId,
        conversationId=request.conversationId,
        domain=request.domain
    )

    return result