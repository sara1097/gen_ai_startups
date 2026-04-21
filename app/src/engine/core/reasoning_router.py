import json
from typing import Dict, List
import pandas as pd
from app.src.chat_schemas.response_schema import ChatResponse, IntentSchema
from app.src.engine.core.intent_classification import (
    classify_intent,
    detect_language, 
    extract_problem_and_requirements
)
from app.src.engine.rag.retriver import retrieve_topk
from app.src.prompt_Engineering.tamplates import FULL_IDEA_TEMPLATE, build_follow_up_prompt, build_general_chat_prompt, build_new_idea_prompt
from app.src.llm.groq_provider import groq_provider
import logging

logger = logging.getLogger(__name__)

llm_provider = groq_provider()


def route_reasoning(
    user_input: str, 
    data: Dict, 
    domain: List[str], 
    isNewConversation: bool,
    conversationId: int,
    clientMessageId: str
) -> Dict:
    
    try:
        logger.debug(f"/nProcessing user input: {user_input}")
        
        # Step 1: Detect language
        lang = detect_language(user_input)
        logger.debug(f"Detected language: {lang}")
        
        # Initialize variables
        structured_data = None
        new_data = None
        inspired_by = None
        context = ""
        extracted = {}
        
        # Step 2: Detect intents
        logger.info(f"Detecting intents...")
        intents_response = classify_intent(user_input)
        logger.debug(f"Detected intents: {intents_response['detected_intents']}")
        
        # Step 3: Extract problem and requirements
        logger.info(f"Extracting problem and requirements...")
        if intents_response["primary_intent"] == "random_solution":
            # Filter problems by domain
            try:
                df = pd.read_excel('data/raw/Problems.xlsx')
                random_domain_based_problem = df[
                    df['problem_sector'].str.lower().isin([d.lower() for d in domain])
                ].sample(n=1)['problem_description'].values[0]
                
                extracted = extract_problem_and_requirements(random_domain_based_problem)
                logger.info(f"Random domain-based problem selected: {random_domain_based_problem}")
                logger.debug(f"Extracted data: {extracted}")

                
            except Exception as e:
                logger.warning(f"Error reading problems file: {e}")
                extracted = extract_problem_and_requirements(user_input)
        elif intents_response["primary_intent"] == "problem_solving":
            extracted = extract_problem_and_requirements(user_input)
            logger.debug(f"Extracted data: {extracted}")

        
        # Step 4: Get context from retriever layer (RAG)
        logger.info(f"Retrieving context from knowledge base...")
        
        def make_context_cards(points):
            """Format retrieved points into readable context cards"""
            logger.info("Making context cards")
            cards = []
            
            if not points:
                logger.warning("No context cards retrieved")
                return ""
            
            for i, p in enumerate(points, 1):
                pl = p.payload or {}
                card = f"""[{i}]
name: {pl.get("name", "N/A")}
domain: {pl.get("domain", "N/A")}
use_case: {pl.get("use_case", "N/A")}
solution: {pl.get("solution", "N/A")}
link: {pl.get("link", "") or pl.get("site", "")}"""
                cards.append(card.strip())
            
            return "\n\n".join(cards)
        
        # Retrieve similar problems/solutions
        if intents_response["primary_intent"] == "problem_solving" or intents_response["primary_intent"] == "random_solution":
            points = retrieve_topk(
                problem_text=extracted.get('core_problem', 'Problem not clearly specified'),
                sectors=domain
            )
        
            context = make_context_cards(points)
            inspired_by = [point.payload.get("name", "") for point in points] if points else None
            
            logger.info(f"Inspired by: {inspired_by}")
            logger.debug(f"Context:\n{context[:200]}...")
        
        # Step 5: Generate or retrieve idea data
        logger.info(f"Determining idea generation strategy...")
        primary_intent = intents_response['primary_intent']
        
        if primary_intent in ["problem_solving", "random_solution"]:
            # Generate new idea
            logger.info(f"Generating new startup idea...")
            core_problem = extracted.get('core_problem', 'Problem not clearly specified')
            
            new_data = llm_provider.generate([
                {"role": "user", "content": FULL_IDEA_TEMPLATE.format(core_problem=core_problem)}
            ])
            logger.debug(f"New idea generated")
        
        elif primary_intent == "alternative_idea":
            # Generate alternative idea for same problem
            logger.info(f"Generating alternative startup idea...")
            problem = data.get(
                'problem_description'
            )
            
            new_data = llm_provider.generate([
                {"role": "user", "content": FULL_IDEA_TEMPLATE.format(core_problem=problem)}
            ])
            logger.debug(f"Alternative idea generated")
        
        else:
            # Use existing data (follow_up, details, feasibility, etc.)
            logger.info(f"Using existing idea data...")
            new_data = data
        
        # Step 6: Parse structured data
        logger.info(f"Parsing structured data...")
        if new_data:
            try:
                structured_data = json.loads(new_data) if isinstance(new_data, str) else new_data
                logger.info(f"Structured data parsed successfully")
            
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse structured data: {e}")
                structured_data = {
                    "raw_text": str(new_data)[:500],
                    "parse_error": str(e)
                }
        else:
            logger.warning(f"No idea data generated")
            structured_data = {
                "raw_text": "No idea data generated"
            }
        
        # Add inspiration sources to structured data
        if inspired_by is not None:
            structured_data["inspired_by"] = json.dumps(inspired_by) if isinstance(inspired_by, list) else str(inspired_by)
        else:
            structured_data["inspired_by"] = "None"
        
        # Step 7: Build unified prompt
        def build_prompt_of_intent(intent):
            if primary_intent in ["problem_solving", "random_solution", "alternative_idea"]:
                return build_new_idea_prompt(
                primary_intent=primary_intent,
                idea_data=structured_data,
                lang=lang
                )
            elif primary_intent == "follow_up":
                return build_follow_up_prompt(
                primary_intent=primary_intent,
                user_input=user_input,
                idea_data=data,
                lang=lang
                )
            elif primary_intent == "general_chat":
                return build_general_chat_prompt(
                user_input=user_input,
                lang=lang
                )
        logger.info(f"Building unified prompt...")
        
        final_prompt = build_prompt_of_intent(primary_intent)
        logger.debug(f"Unified prompt built")
        
        # Step 8: Generate response content
        logger.info(f"Generating response content...")
        
        content = llm_provider.generate([
            {"role": "user", "content": final_prompt}
        ])
        
        logger.info(f"Response content generated")
        
        # Step 9: Determine response type
        is_idea = primary_intent in ["problem_solving", "random_solution", "alternative_idea"]
        
        # Step 10: Build response object
        logger.info(f"Building response object...")
        
        response_data = {
            "content": content,
            "conversationId": conversationId,
            "clientMessageId": clientMessageId,
            "role": "ai",
            "is_idea_saved": False,
            "is_full_idea": is_idea,
            "data": structured_data
        }
        
        # Add conversation title for new conversations
        if isNewConversation:
            conversation_title = extracted.get('core_problem', 'New Conversation')
            response_data["conversation_title"] = conversation_title
            logger.info(f"New conversation titled: {conversation_title}")
        
        # Validate and return response
        logger.info(f"Validating response...")
        response = ChatResponse(**response_data)
        
        logger.info(f"Response ready to send")
        logger.debug(f"Response summary - Intent: {primary_intent}, Is Idea: {is_idea}")
        
        return response.dict()
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise
    
    except KeyError as e:
        logger.error(f"Missing required field: {e}")
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error in route_reasoning: {type(e).__name__}")
        logger.exception(e)
        raise