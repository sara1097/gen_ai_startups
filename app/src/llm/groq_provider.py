import os
from groq import Groq
from app.src.llm.base import BaseLLM

from dotenv import load_dotenv
load_dotenv(".env")

class groq_provider(BaseLLM):
  def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

  def generate(self, messages: list) -> str:
        """
        Generate response from Groq
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages, 
            temperature=0.2,
        )
        return response.choices[0].message.content
    



  def stream(self , message: list):
   stream = self.client.chat.completions.create(
       model="llama-3.1-8b-instant",
    messages = message,
    stream=True
  )
   return stream

  