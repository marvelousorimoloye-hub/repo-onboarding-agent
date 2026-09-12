"""
Model client wrappers. Every agent calls `get_client_for_role(role)` rather
than instantiating a provider SDK directly — this is what makes the prod
model swap a config change instead of a code change.

Phase 0 — fully implemented.
"""
from abc import ABC, abstractmethod

from groq import Groq
from google import genai

from config.settings import AgentRole, ModelProvider, KEYS, get_model_provider, get_model_name


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        ...


class GroqClient(LLMClient):
    def __init__(self, model_name: str):
        self._client = Groq(api_key=KEYS.groq_api_key)
        self._model_name = model_name

    def complete(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        )
        return response.choices[0].message.content


class GeminiClient(LLMClient):
    def __init__(self, model_name: str):
        self._client = genai.Client(api_key=KEYS.gemini_api_key)
        self._model_name = model_name

    def complete(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=user_prompt,
            config={"system_instruction": system_prompt, **kwargs},
        )
        return response.text


_CLIENT_CLASSES: dict[ModelProvider, type[LLMClient]] = {
    ModelProvider.GROQ: GroqClient,
    ModelProvider.GEMINI: GeminiClient,
}

# Simple per-process cache so repeated calls for the same role reuse one
# client instance instead of reconnecting every call.
_client_cache: dict[AgentRole, LLMClient] = {}


def get_client_for_role(role: AgentRole) -> LLMClient:
    if role not in _client_cache:
        provider = get_model_provider(role)
        model_name = get_model_name(role)
        _client_cache[role] = _CLIENT_CLASSES[provider](model_name)
    return _client_cache[role]