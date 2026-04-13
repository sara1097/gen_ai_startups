from dotenv import load_dotenv
load_dotenv(".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.src.chat_schemas.response_schema import ChatRequest, ChatResponse
from app.src.engine.core.reasoning_router import route_reasoning
from app.src.engine.core.logger import setup_logging

setup_logging()

app = FastAPI(title="Startup AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    result = route_reasoning(
        user_input=request.content,
        data=request.data,
        isNewConversation=request.isNewConversation,
        conversationId=request.conversationId,
        domain=request.domain
    )
    return result