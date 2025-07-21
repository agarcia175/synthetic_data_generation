"""
A base minimum client to wrap any other actual LLM client, local or external.
Since the only expected need is to ask for certain generation, I sacrifice any extra (unnecessary) flexibility for simplicity.
Just a method to ask for the generation, plus some minimal generation parameters (usual ones: temperature, max_tokens, ...)
"""
from abc import ABC, abstractmethod
from typing import Coroutine, Any

from ollama import ChatResponse, AsyncClient, Client


# hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M

class LLMClientBase(ABC):

    @abstractmethod
    def generate(self, system_message, user_message, temperature: float, max_tokens: int) -> str:
        pass

    @abstractmethod
    def generate_async(self, system_message, user_message, temperature: float, max_tokens: int) -> Coroutine[Any, Any, str]:
        pass

    @classmethod
    def get_system_role(cls):
        return "system"

    @classmethod
    def get_user_role(cls):
        return "user"


class OllamaLLMClient(LLMClientBase):

    def __init__(self, host: str = None, model_name: str = 'hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M') -> None:
        self.model_name = model_name
        # self.client = Client(base_url=host)
        self.client = Client()
        self.async_client = AsyncClient()

    def generate(self, system_message, user_message, temperature: float, max_tokens: int, think: bool = False) -> str:
        messages = []
        if system_message:
            messages.append({'role': self.get_system_role(), 'content': system_message})
        messages.append({'role': self.get_user_role(), 'content': user_message})
        # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
        response: ChatResponse = self.client.chat(model=self.model_name, messages=messages, think=think, options={'temperature': temperature, 'num_predict': max_tokens})
        generated_content = response['message']['content']
        return generated_content

    async def generate_async(self, system_message, user_message, temperature: float, max_tokens: int, think: bool = False) -> Coroutine[Any, Any, str]:
        messages = []
        if system_message:
            messages.append({'role': self.get_system_role(), 'content': system_message})
        messages.append({'role': self.get_user_role(), 'content': user_message})
        # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
        response: ChatResponse = await self.async_client.chat(model=self.model_name, messages=messages, think=think, options={'temperature': temperature, 'num_predict': max_tokens})
        generated_content = response['message']['content']
        return generated_content
