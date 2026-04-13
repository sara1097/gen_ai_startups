import json
from typing import Dict, List
import pandas as pd
from app.src.chat_schemas.response_schema import ChatResponse, IntentSchema
from app.src.engine.core.intent_classification import (
    classify_intent, 
    extract_problem_and_requirements
)
from app.src.engine.rag.retriver import retrieve_topk
from app.src.prompt_Engineering.tamplates import FULL_IDEA_TEMPLATE
from app.src.prompt_Engineering.tamplates import build_unified_prompt
from app.src.llm.groq_provider import groq_provider
import logging

logger = logging.getLogger(__name__)

llm_provider = groq_provider()

def route_reasoning(
    user_input: str, 
    data: Dict, 
    domain: str, 
    isNewConversation: bool,
    conversationId: str
) -> Dict:
   
    
    logger.debug(f"\nProcessing user input: {user_input}")
    
    structured_data = None
    new_data = None
    
    # Step 1: Detect intents
    intents_response = classify_intent(user_input)
    logger.debug(f"Detected intents: {intents_response['detected_intents']}")
    
    # Step 2: Extract problem and requirements
    if intents_response["primary_intent"] == "random_solution":
        # Filter problems by domain
        try:
            df = pd.read_excel('data/raw/Problems.xlsx')
            random_domain_based_problem = df[
                df['problem_sector'].str.lower() == domain.lower()
            ].sample(n=1)['problem_description'].values[0]
            extracted = extract_problem_and_requirements(random_domain_based_problem)
            print(f"Random domain based problem: {random_domain_based_problem}\n")
        except Exception as e:
            print(f"Error reading problems: {e}")
            extracted = extract_problem_and_requirements(user_input)
    else:
        extracted = extract_problem_and_requirements(user_input)
    
    logger.debug(f"Extracted data: {extracted}")
    
    # Step 3: Get context from retriever layer
    def make_context_cards(points):
     logger.info("Making The Context Cards")
     cards = []
     for i, p in enumerate(points, 1):
        pl = p.payload or {}
        cards.append(f"""[{i}]
        name: {pl.get("name","")}
        domain: {pl.get("domain","")}
        use_case: {pl.get("use_case","")}
        solution: {pl.get("solution","")}
        link: {pl.get("link","") or pl.get("site","")}""".strip())
     return "\n\n".join(cards)
     
    
    points = retrieve_topk(
        problem_text=extracted.get('core_problem', 'Problem not clearly specified'),
        sector=domain
    )
    context = make_context_cards(points)
    inspired_by = [point.payload.get("name","") for point in points] if points else None

    logger.info(f"Inspired by: {inspired_by}")

    logger.debug(f"The Context {context}")
    
    # Step 4: Generate or retrieve idea data
    primary_intent = intents_response['primary_intent']
    
    if primary_intent in ["problem_solving", "random_solution"]:
        # Generate new idea
        logger.debug(f"Generating new startup idea...")
        core_problem = extracted.get('core_problem', 'Problem not clearly specified')
        new_data = llm_provider.generate([
            {"role": "user", "content": FULL_IDEA_TEMPLATE.format(core_problem=core_problem)}
        ])
    
    elif primary_intent == "alternative_idea":
        logger.info(f"Generating alternative startup idea...")
        problem = data.get('problem_description', extracted.get('core_problem', 'Problem not clearly specified'))
        new_data = llm_provider.generate([
            {"role": "user", "content": FULL_IDEA_TEMPLATE.format(core_problem=problem)}
        ])
    
    else:
        logger.debug(f"Using existing idea data...")
        new_data = data
    
    if new_data:
        try:
            structured_data = json.loads(new_data) if isinstance(new_data, str) else new_data
            logger.info(f"Structured data parsed")
        except (json.JSONDecodeError, TypeError) as e:
            logger.exception(f"Failed to parse structured data: {e}")
            structured_data = {
                "raw_text": str(new_data),
                "parse_error": str(e)
            }
    else:
        structured_data = {
            "raw_text": "No idea data generated"
        }
    
    # Step 5: Build unified prompt (to generate the response's content)
    logger.info(f"Building unified prompt...")
    
    final_prompt = build_unified_prompt(
        detected_intents=intents_response['detected_intents'],
        extracted_data=extracted,
        context=context,
        primary_intent=primary_intent,
        idea_data=structured_data
    )
    
    # Step 6: Call LLM with the final prompt to generate response
    logger.info(f"Generating response...")
    
    content = llm_provider.generate([
        {"role": "user", "content": final_prompt}
    ])
    
    logger.info(f"Response received")
    
    # Determine if this is an idea response
    is_idea = primary_intent in ["problem_solving", "random_solution", "alternative_idea"]
    
    # Step 7: Return response
    if isNewConversation:
        conversation_title = extracted.get('core_problem', 'New Conversation')
        
        return ChatResponse(
            content=content,
            conversationId=conversationId,
            conversation_title=conversation_title,
            role='ai',
            is_idea_saved=False,
            is_full_idea=is_idea,
            data=structured_data,
            inspired_by= inspired_by
        ).dict()
    
    else:
        return ChatResponse(
            content=content,
            conversationId=conversationId,
            role='ai',
            is_idea_saved=False,
            is_full_idea=is_idea,
            data=structured_data,
            inspired_by= inspired_by
        ).dict()