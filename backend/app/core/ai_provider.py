import os
import logging
import aiohttp
import json
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class AIProviderBase(ABC):
    """Base class for AI providers"""
    
    @abstractmethod
    async def generate_answer(self, question: str, context: str) -> str:
        pass

class GroqProvider(AIProviderBase):
    """Groq AI provider implementation"""
    
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
    async def generate_answer(self, question: str, context: str) -> str:
        try:
            prompt = f"""You are a helpful AI assistant for Confluence documentation. 
Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful Confluence documentation assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1024
                }
                
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['choices'][0]['message']['content']
                    else:
                        error_text = await response.text()
                        logger.error(f"Groq API error: {response.status} - {error_text}")
                        return f"Sorry, I encountered an error: {response.status}"
                        
        except Exception as e:
            logger.error(f"Error in GroqProvider: {str(e)}")
            return "I'm having trouble connecting to the AI service. Please try again."

class OpenAIProvider(AIProviderBase):
    """OpenAI provider implementation"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
    async def generate_answer(self, question: str, context: str) -> str:
        try:
            prompt = f"""You are a helpful AI assistant for Confluence documentation. 
Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful Confluence documentation assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1024
                }
                
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['choices'][0]['message']['content']
                    else:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error: {response.status} - {error_text}")
                        return f"Sorry, I encountered an error: {response.status}"
                        
        except Exception as e:
            logger.error(f"Error in OpenAIProvider: {str(e)}")
            return "I'm having trouble connecting to the AI service. Please try again."

# This is the key part - the class that main.py imports
class AIProvider:
    """Main AI Provider factory class"""
    
    def __init__(self):
        provider_type = os.getenv("AI_PROVIDER", "groq").lower()
        
        if provider_type == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            model = os.getenv("AI_MODEL", "llama-3.1-70b-versatile")
            
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
            
            logger.info(f"[AIProvider] Initializing Groq with model: {model}")
            self._provider = GroqProvider(api_key, model)
            
        elif provider_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            model = os.getenv("AI_MODEL", "gpt-4")
            
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            logger.info(f"[AIProvider] Initializing OpenAI with model: {model}")
            self._provider = OpenAIProvider(api_key, model)
            
        else:
            raise ValueError(f"Unknown AI provider: {provider_type}")
    
    async def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using the configured provider"""
        if not context:
            return "I don't have enough context to answer that question. Please sync Confluence data first."
        
        return await self._provider.generate_answer(question, context)