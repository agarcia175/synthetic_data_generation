"""
A base minimum client to wrap any other actual LLM client, local or external.
Since the only expected need is to ask for certain generation, I sacrifice any extra (unnecessary) flexibility for simplicity.
Just a method to ask for the generation, plus some minimal generation parameters (usual ones: temperature, max_tokens, ...)
"""
from abc import ABC, abstractmethod
from typing import Coroutine, Any


# hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M

class LLMClientBase(ABC):

    @abstractmethod
    def generate(self, system_message, user_message, temperature: float, max_tokens: int) -> str:
        pass

    @abstractmethod
    def generate_async(self, system_message, user_message, temperature: float, max_tokens: int) -> Coroutine[
        Any, Any, str]:
        pass

    @classmethod
    def get_system_role(cls):
        return "system"

    @classmethod
    def get_user_role(cls):
        return "user"
