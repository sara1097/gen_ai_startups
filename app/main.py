from app.src.engine.core.logger import setup_logging
setup_logging()

import logging
from fastapi import FastAPI, Request
from dotenv import load_dotenv

from app.src.chat_schemas.response_schema import ChatRequest, ChatResponse
from app.src.engine.core.reasoning_router import route_reasoning


load_dotenv(".env")

app = FastAPI(title="Startup AI Service")

logger = logging.getLogger(__name__)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: Request):
    
    try:
        body = await request.json()
    except Exception:
        raw_body = await request.body()
        logger.error("Request is not valid JSON")
        logger.error(f"Raw body: {raw_body.decode(errors='ignore')}")
        raise

    logger.info("New request received")
    logger.debug(f"Body: {str(body)[:500]}")
    logger.debug(f"Headers: {dict(request.headers)}")

    try:
        parsed_request = ChatRequest(**body)
    except Exception as e:
        logger.error("Validation error")
        logger.error(f"Error: {e}")
        logger.error(f"Bad body: {body}")
        raise

    try:
        logger.info("Starting reasoning...")

        result = route_reasoning(
            user_input=parsed_request.content,
            data=parsed_request.data,
            isNewConversation=parsed_request.isNewConversation,
            clientMessageId=parsed_request.clientMessageId,
            conversationId=parsed_request.conversationId,
            domain=parsed_request.domain
        )

        logger.debug(f"response {result}")

        logger.info("Response generated successfully")

    except Exception as e:
        logger.error("Error in reasoning")
        logger.exception(e)
        raise

    return result