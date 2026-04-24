# app/src/llm/groq_provider.py
import logging
import time
from typing import List, Dict, Optional
from groq import Groq, APIError, RateLimitError

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

class GroqProvider:
    """
    Optimized provider for Qwen models on Groq.
    Handles <think> blocks and implements retry logic.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.client = Groq(api_key=self.settings.GROQ_API_KEY)
        self.max_retries = 3
        self.model_name = self.settings.LLM_MODEL
        
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.2,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate response with Qwen-specific optimizations.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Controls randomness (0.2 for structured tasks)
            max_tokens: Maximum tokens in response
            
        Returns:
            Raw response text (may contain <think> blocks)
        """
        retries = 0
        
        while retries <= self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                content = response.choices[0].message.content
                logger.debug(f"Qwen response received ({len(content)} chars)")
                return content
                
            except RateLimitError:
                wait_time = 2 ** retries
                logger.warning(
                    f"Rate limit hit. Retrying in {wait_time}s... "
                    f"({retries}/{self.max_retries})"
                )
                time.sleep(wait_time)
                retries += 1
                
            except APIError as e:
                logger.error(f"Groq API Error: {e}")
                retries += 1
                if retries <= self.max_retries:
                    time.sleep(1)
                    
            except Exception as e:
                logger.exception(f"Unexpected error during LLM generation: {e}")
                raise RuntimeError(f"LLM generation failed: {e}")
                
        raise RuntimeError(
            f"Failed to generate response after {self.max_retries} retries."
        )
    
    def generate_structured(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.1
    ) -> str:
        """
        Generate structured JSON output (lower temperature for consistency).
        """
        return self.generate(messages, temperature=temperature, max_tokens=4096)