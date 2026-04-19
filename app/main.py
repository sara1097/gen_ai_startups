from app.src.engine.core.logger import setup_logging
setup_logging()

import logging
from fastapi import FastAPI, HTTPException, Body
from dotenv import load_dotenv
from app.src.chat_schemas.response_schema import ChatRequest, ChatResponse
from app.src.engine.core.reasoning_router import route_reasoning

load_dotenv(".env")

app = FastAPI(
    title="Startup Generator AI Service",
    description="AI-powered startup idea generator for MENA region",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

logger = logging.getLogger(__name__)


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Generate startup ideas or respond to chat",
    tags=["Chat"],
    responses={
        200: {"description": "Success - Response generated"},
        400: {"description": "Bad request - Invalid input"},
        500: {"description": "Internal server error"}
    }
)
async def chat_endpoint(
    request: ChatRequest = Body(
        ...,
        example={
            "content": "I want to solve expensive education in Egypt",
            "conversationId": 123,
            "clientMessageId": "msg_001",
            "isNewConversation": True,
            "domain": ["education"],
            "data": None
        }
    )
):
    
    try:
        logger.info(f"New chat request received")
        logger.debug(f"Content: {request.content[:100]}...")
        logger.debug(f"Conversation ID: {request.conversationId}")
        logger.debug(f"Domain: {request.domain}")
        logger.debug(f"Is New Conversation: {request.isNewConversation}")
        
        logger.info("Starting reasoning engine...")
        result = route_reasoning(
            user_input=request.content,
            data=request.data or {},
            isNewConversation=request.isNewConversation,
            clientMessageId=request.clientMessageId,
            conversationId=request.conversationId,
            domain=request.domain or []
        )
        
        logger.info("Response generated successfully")
        logger.debug(f"Is Full Idea: {result.get('is_full_idea')}")
        
        result['clientMessageId'] = request.clientMessageId
        
        response = ChatResponse(**result)
        
        logger.info(f"Response sent to client (Conversation: {request.conversationId})")
        return response
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {str(e)}"
        )
    
    except KeyError as e:
        logger.error(f"Missing required field: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Missing required field: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {type(e).__name__}")
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get(
    "/health",
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check():

    logger.debug("Health check requested")
    return {
        "status": "healthy",
        "service": "Startup Generator AI Service"
    }


@app.get(
    "/",
    tags=["Info"],
    summary="API Information"
)
async def root():

    return {
        "name": "Startup Generator AI Service",
        "version": "1.0.0",
        "description": "AI-powered startup idea generator for MENA region",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health": "/health"
    }


@app.get(
    "/docs",
    tags=["Documentation"],
    summary="Interactive API Documentation"
)
async def swagger_ui():
    """Redirect to Swagger UI documentation"""
    return {"redirect": "/docs"}

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return {
        "error": True,
        "status_code": exc.status_code,
        "detail": exc.detail
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled Exception: {type(exc).__name__} - {str(exc)}")
    return {
        "error": True,
        "status_code": 500,
        "detail": "Internal server error"
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Startup Generator AI Service...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )