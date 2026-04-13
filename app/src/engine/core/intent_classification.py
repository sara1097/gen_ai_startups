import json
import re
from typing import Dict
from app.src.llm.groq_provider import groq_provider
from app.src.prompt_Engineering.tamplates import INTENTS_DETECTION_TEMPLATE
import logging

logger = logging.getLogger(__name__)

llm_provider = groq_provider()

def classify_intent(message: str) -> Dict:
    """
    Classify user intent from message using LLM
    """
    logger.info("Classifying intent")
    try:
        response = llm_provider.generate([
            {"role": "user", "content": INTENTS_DETECTION_TEMPLATE.format(user_message=message)}
        ])
        
        logger.debug(f"Raw LLM response: {response}")

        cleaned_response = clean_json_response(response)
        parsed = json.loads(cleaned_response)
        
        logger.info("Intent classification succeeded")

        return parsed
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed in classify_intent: {e}")
        return get_default_intent(message)
    
    except Exception as e:
        logger.exception(f"Unexpected error in classify_intent: {type(e).__name__}: {e}")
        return get_default_intent(message)


def extract_problem_and_requirements(user_input: str) -> Dict:

    logger.info("Extracting problem and requirements") 

    extraction_prompt = f"""Extract information from this input. Return ONLY valid JSON.

User input: "{user_input}"

Return this exact JSON format (no other text):
{{"core_problem": "", "requirements": [], "references_previous": false, "questions": [], "constraints": []}}

Fill the fields based on the user input. If a field is empty, use empty string or empty list."""
    
    try:
        response = llm_provider.generate([
            {"role": "user", "content": extraction_prompt}
        ])
        
        logger.debug(f"Raw extraction response: {response}")

        cleaned_response = extract_json_only(response)
        
        parsed = json.loads(cleaned_response)
        
        logger.info("Extraction succeeded")

        return {
            "core_problem": parsed.get("core_problem", ""),
            "requirements": parsed.get("requirements", []),
            "references_previous": parsed.get("references_previous", False),
            "questions": parsed.get("questions", []),
            "constraints": parsed.get("constraints", [])
        }
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON Parse Error: {e}")
        return get_default_extraction(user_input)
    
    except Exception as e:
        logger.exception(f"Error: {type(e).__name__}: {e}")
        return get_default_extraction(user_input)


def extract_json_only(text: str) -> str:
    """
    Extract ONLY the first valid JSON object from text
    """
    import re
    
    # Remove markdown
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Find first '{'
    start = text.find('{')
    if start == -1:
        return '{}'
    
    # Count braces to find matching '}'
    count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            count += 1
        elif text[i] == '}':
            count -= 1
            if count == 0:
                return text[start:i+1]
    
    return '{}'

def clean_json_response(response: str) -> str:
    """
    Clean LLM response by removing markdown and extra text
    """
    import re
    
    # Remove markdown code block markers
    response = re.sub(r'```(?:json|python|text)?\s*\n?', '', response)
    response = re.sub(r'\n?```', '', response)
    
    # Remove any text before first '{'
    json_start = response.find('{')
    if json_start != -1:
        response = response[json_start:]
    
    # Remove any text after last '}'
    json_end = response.rfind('}')
    if json_end != -1:
        response = response[:json_end + 1]
    
    return response.strip()


def get_default_intent(user_input: str) -> Dict:
    """
    Return default intent when LLM parsing fails
    """
    return {
        "detected_intents": [
            {
                "intent": "general_chat",
                "confidence": "high",
                "relevant_text": user_input,
                "priority": 1
            }
        ],
        "primary_intent": "general_chat",
        "secondary_intents": []
    }


def get_default_extraction(user_input: str) -> Dict:
    """
    Return default extraction when parsing fails
    """
    return {
        "core_problem": "",
        "requirements": [],
        "references_previous": False,
        "questions": [user_input],
        "constraints": []
    }