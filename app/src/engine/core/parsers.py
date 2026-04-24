# app/src/engine/core/parsers.py
import json
import re
import logging
from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseModel)

class QwenParser:
    """Parser specifically optimized for Qwen model outputs."""
    
    @staticmethod
    def remove_think_blocks(text: str) -> str:
        """Remove <think>...</think> blocks from Qwen output."""
        # Remove <think>...</think> tags (greedy)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Remove any remaining <think> or </think> tags
        text = re.sub(r'</?think>', '', text)
        
        return text.strip()
    
    @staticmethod
    def remove_markdown(text: str) -> str:
        """Remove markdown formatting."""
        # Remove markdown code blocks
        text = re.sub(r'```(?:json|python|text)?\s*\n?', '', text)
        text = re.sub(r'\n?```', '', text)
        
        return text.strip()
    
    @staticmethod
    def extract_json(text: str) -> str:
        """Extract JSON object or array from text."""
        # Remove think blocks first
        text = QwenParser.remove_think_blocks(text)
        text = QwenParser.remove_markdown(text)
        
        # Find JSON object
        obj_start = text.find('{')
        obj_end = text.rfind('}')
        
        # Find JSON array
        arr_start = text.find('[')
        arr_end = text.rfind(']')
        
        # Determine which to extract
        if obj_start != -1 and (arr_start == -1 or obj_start < arr_start):
            if obj_end != -1:
                return text[obj_start:obj_end + 1]
        elif arr_start != -1 and arr_end != -1:
            return text[arr_start:arr_end + 1]
        
        # If no JSON found, return original (will fail validation)
        return text
    
    @classmethod
    def parse_and_validate(cls, response_text: str, schema: Type[T]) -> T:
        """
        Parse Qwen response and validate against schema.
        
        Args:
            response_text: Raw response from Qwen (may contain <think> blocks)
            schema: Pydantic model to validate against
            
        Returns:
            Validated Pydantic model instance
            
        Raises:
            ValueError: If parsing or validation fails
        """
        try:
            json_text = cls.extract_json(response_text)
            parsed_json = json.loads(json_text)
            return schema(**parsed_json)
            
        except json.JSONDecodeError as e:
            logger.error(
                f"JSON parsing failed.\n"
                f"Raw text: {response_text[:200]}...\n"
                f"Error: {e}"
            )
            raise ValueError(f"Failed to parse response as JSON: {e}")
            
        except ValidationError as e:
            logger.error(f"Schema validation failed. Errors: {e.errors()}")
            raise ValueError(f"Response did not match expected schema: {e}")
