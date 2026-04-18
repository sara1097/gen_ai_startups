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
            "domain": "education",
            "data": None
        }
    )
):
    """
    Process user input and generate startup ideas or responses.
    
    **Request Parameters:**
    - **content**: The user's message, problem description, or question
    - **conversationId**: Unique integer identifier for this conversation
    - **clientMessageId**: Unique string identifier for this specific message
    - **isNewConversation**: Boolean flag indicating if this is a new conversation
    - **domain**: Business domain/sector (education, healthcare, technology, etc.)
    - **data**: Previous idea data for follow-up or alternative requests (optional)
    
    **Response Fields:**
    - **content**: The AI-generated response message
    - **conversationId**: Echo of the conversation ID
    - **clientMessageId**: Echo of the message ID for tracking
    - **conversation_title**: Generated title for new conversations
    - **role**: Always "ai" for AI responses
    - **is_idea_saved**: Whether the idea was saved (for frontend state)
    - **is_full_idea**: Boolean indicating if response contains a complete startup idea
    - **data**: Full structured idea data (when is_full_idea=true)
    - **inspired_by**: List of inspiration sources
    
    **Example Intents:**
    - Problem solving: "I want to solve expensive education in Egypt"
    - Random solution: "Give me a startup idea"
    - Follow-up: "Tell me more about the business model"
    - Alternative: "Give me a different approach"
    - Details: "Explain the implementation steps"
    - Feasibility: "Is this feasible?"
    - Novelty: "Is this innovative?"
    - General chat: "What's trending in startups?"
    """
    
    try:
        # Log incoming request
        logger.info(f"📨 New chat request received")
        logger.debug(f"Content: {request.content[:100]}...")
        logger.debug(f"Conversation ID: {request.conversationId}")
        logger.debug(f"Domain: {request.domain}")
        logger.debug(f"Is New Conversation: {request.isNewConversation}")
        
        # Call reasoning router
        logger.info("🔄 Starting reasoning engine...")
        result = route_reasoning(
            user_input=request.content,
            data=request.data or {},
            isNewConversation=request.isNewConversation,
            clientMessageId=request.clientMessageId,
            conversationId=request.conversationId,
            domain=request.domain or "general"
        )
        
        # Validate response structure
        logger.info("✅ Response generated successfully")
        logger.debug(f"Is Full Idea: {result.get('is_full_idea')}")
        
        # Ensure clientMessageId is in response
        result['clientMessageId'] = request.clientMessageId
        
        # Create validated response
        response = ChatResponse(**result)
        
        logger.info(f"✅ Response sent to client (Conversation: {request.conversationId})")
        return response
    
    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {str(e)}"
        )
    
    except KeyError as e:
        logger.error(f"❌ Missing required field: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Missing required field: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"❌ Unexpected error in chat endpoint: {type(e).__name__}")
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
    """
    Check if the API is running and healthy.
    
    **Returns:**
    - **status**: "healthy" if the API is running
    """
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
    """
    Get API information and documentation links.
    
    **Returns:**
    - **name**: Service name
    - **version**: API version
    - **docs**: Link to interactive API documentation
    - **openapi**: Link to OpenAPI schema
    """
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


# Error handlers
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
    """Custom general exception handler"""
    logger.error(f"Unhandled Exception: {type(exc).__name__} - {str(exc)}")
    return {
        "error": True,
        "status_code": 500,
        "detail": "Internal server error"
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Startup Generator AI Service...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )