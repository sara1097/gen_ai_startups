from app.src.engine.core.logger import setup_logging

setup_logging()

from dotenv import load_dotenv

from app.src.engine.core.reasoning_router import route_reasoning
load_dotenv(".env")

from app.src.chat_schemas.response_schema import ChatRequest, ChatResponse
from fastapi import FastAPI
from fastapi import Request

from app.src.engine.core.reasoning_router import route_reasoning

app = FastAPI(title="Startup AI Service")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: Request):
    body = await request.json()

    logger.info("📥 Raw request received")
    logger.debug(f"Full body: {body}")

    # بعد كده تحولي لـ schema
    parsed_request = ChatRequest(**body)

    result = route_reasoning(
        user_input=parsed_request.content,
        data=parsed_request.data,
        isNewConversation=parsed_request.isNewConversation,
        clientMessageId=parsed_request.clientMessageId,
        conversationId=parsed_request.conversationId,
        domain=parsed_request.domain
    )

    return result