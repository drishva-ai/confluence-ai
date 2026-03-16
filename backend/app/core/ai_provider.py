"""
AI Provider Abstraction Layer
==============================
This is the heart of the model-agnostic design.
To switch AI models: change AI_PROVIDER in .env — nothing else.

Supported providers:
  - groq      → Llama 3.1 70B (FREE — recommended for start)
  - openai    → GPT-4o (best quality, paid)
  - anthropic → Claude 3.5 Sonnet (excellent, paid)
  - google    → Gemini 1.5 Pro (good, paid)
  - ollama    → Any local model (fully free, needs GPU)
"""

import os
from typing import AsyncGenerator
from dotenv import load_dotenv

load_dotenv()

AI_PROVIDER = os.getenv("AI_PROVIDER", "groq").lower()
AI_MODEL    = os.getenv("AI_MODEL", "llama-3.1-70b-versatile")


# ── Base interface all providers must implement ──────────────────────────────
class AIProviderBase:
    """Every AI provider implements these two methods. Nothing else."""

    async def chat(self, system_prompt: str, user_message: str, context: str) -> str:
        """Send a message and get a response."""
        raise NotImplementedError

    async def stream_chat(
        self, system_prompt: str, user_message: str, context: str
    ) -> AsyncGenerator[str, None]:
        """Stream response token by token for real-time UI."""
        raise NotImplementedError
        yield  # makes it a generator


# ── GROQ (Free Llama) ────────────────────────────────────────────────────────
class GroqProvider(AIProviderBase):
    """
    Groq runs Llama 3.1 70B in the cloud.
    FREE: 6000 requests/day, extremely fast.
    Get API key: https://console.groq.com
    """

    def __init__(self):
        from groq import AsyncGroq
        self.client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self.model  = AI_MODEL or "llama-3.1-70b-versatile"

    async def chat(self, system_prompt: str, user_message: str, context: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {user_message}"}
            ],
            temperature=0.1,   # low = more factual, less creative
            max_tokens=1024,
        )
        return response.choices[0].message.content

    async def stream_chat(
        self, system_prompt: str, user_message: str, context: str
    ) -> AsyncGenerator[str, None]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {user_message}"}
            ],
            temperature=0.1,
            max_tokens=1024,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ── OPENAI (GPT-4o) ──────────────────────────────────────────────────────────
class OpenAIProvider(AIProviderBase):
    """
    OpenAI GPT-4o — best quality for complex documents.
    Cost: ~$0.01 per 1000 queries (very affordable).
    Get API key: https://platform.openai.com/api-keys
    """

    def __init__(self):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model  = AI_MODEL or "gpt-4o"

    async def chat(self, system_prompt: str, user_message: str, context: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {user_message}"}
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    async def stream_chat(
        self, system_prompt: str, user_message: str, context: str
    ) -> AsyncGenerator[str, None]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {user_message}"}
            ],
            temperature=0.1,
            max_tokens=1024,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ── ANTHROPIC (Claude) ───────────────────────────────────────────────────────
class AnthropicProvider(AIProviderBase):
    """
    Anthropic Claude 3.5 Sonnet — excellent for long documents.
    Get API key: https://console.anthropic.com
    """

    def __init__(self):
        import anthropic
        self.client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model  = AI_MODEL or "claude-3-5-sonnet-20241022"

    async def chat(self, system_prompt: str, user_message: str, context: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_message}"}
            ],
        )
        return response.content[0].text

    async def stream_chat(
        self, system_prompt: str, user_message: str, context: str
    ) -> AsyncGenerator[str, None]:
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_message}"}
            ],
        ) as stream:
            async for text in stream.text_stream:
                yield text


# ── GOOGLE (Gemini) ──────────────────────────────────────────────────────────
class GoogleProvider(AIProviderBase):
    """
    Google Gemini 1.5 Pro — great multimodal capabilities.
    Get API key: https://aistudio.google.com/app/apikey
    """

    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(
            model_name=AI_MODEL or "gemini-1.5-pro",
            system_instruction=None,
        )

    async def chat(self, system_prompt: str, user_message: str, context: str) -> str:
        import asyncio
        prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {user_message}"
        response = await asyncio.to_thread(self.model.generate_content, prompt)
        return response.text

    async def stream_chat(
        self, system_prompt: str, user_message: str, context: str
    ) -> AsyncGenerator[str, None]:
        import asyncio
        prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {user_message}"
        response = await asyncio.to_thread(
            self.model.generate_content, prompt, stream=True
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text


# ── OLLAMA (local, fully free) ───────────────────────────────────────────────
class OllamaProvider(AIProviderBase):
    """
    Ollama runs models locally on your PC — completely free forever.
    Needs: ollama.ai installed + a GPU recommended.
    Run: ollama pull llama3.2  then  ollama serve
    """

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model    = AI_MODEL or "llama3.2"

    async def chat(self, system_prompt: str, user_message: str, context: str) -> str:
        import httpx
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {user_message}"}
                    ],
                }
            )
            return response.json()["message"]["content"]

    async def stream_chat(
        self, system_prompt: str, user_message: str, context: str
    ) -> AsyncGenerator[str, None]:
        import httpx, json
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": True,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {user_message}"}
                    ],
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if data.get("message", {}).get("content"):
                            yield data["message"]["content"]


# ── FACTORY — picks the right provider from .env ─────────────────────────────
def get_ai_provider() -> AIProviderBase:
    """
    THE ONE FUNCTION that reads .env and returns the right provider.
    This is called once at startup — nowhere else in the codebase.
    
    To switch models: change AI_PROVIDER in .env, redeploy.
    """
    providers = {
        "groq":      GroqProvider,
        "openai":    OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google":    GoogleProvider,
        "ollama":    OllamaProvider,
    }

    provider_class = providers.get(AI_PROVIDER)
    if not provider_class:
        raise ValueError(
            f"Unknown AI_PROVIDER: '{AI_PROVIDER}'. "
            f"Choose from: {', '.join(providers.keys())}"
        )

    print(f"[AI] Using provider: {AI_PROVIDER} | model: {AI_MODEL}")
    return provider_class()


# ── System prompt used for ALL providers ─────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful AI assistant for a company's internal knowledge base.
You answer questions based ONLY on the provided context from Confluence pages.

Rules:
1. Answer only from the provided context — never make up information
2. If the answer is not in the context, say "I could not find this in the documentation"
3. Always mention which page or document the answer came from
4. Keep answers clear and concise
5. Use bullet points for lists and steps
6. If the question is about a process, give step-by-step instructions

You are professional, helpful and accurate."""
