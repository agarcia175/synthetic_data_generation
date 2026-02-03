import os
from typing import Coroutine, Any

from openai.types.chat import ChatCompletion

from synthetic_data_gen.llms.llm_client import LLMClientBase
from openai import OpenAI, AsyncOpenAI


class OpenAIClient(LLMClientBase):

    def __init__(self, host: str = None, model_name: str = 'gpt-4.1-nano') -> None:
        self.model_name = model_name
        # self.client = Client(base_url=host)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, system_message, user_message, temperature: float, max_tokens: int, think: bool = False) -> str:
        messages = []
        if system_message:
            messages.append({'role': self.get_system_role(), 'content': system_message})
        messages.append({'role': self.get_user_role(), 'content': user_message})
        # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
        response: ChatCompletion = self.client.chat.completions.create(model=self.model_name, messages=messages,
                                                                       temperature=temperature, max_tokens=max_tokens)
        # generated_content = response['message']['content']
        generated_content = response.choices[0].message.content
        return generated_content

    async def generate_async(self, system_message, user_message, temperature: float, max_tokens: int,
                             think: bool = False) -> Coroutine[Any, Any, str]:
        messages = []
        if system_message:
            messages.append({'role': self.get_system_role(), 'content': system_message})
        messages.append({'role': self.get_user_role(), 'content': user_message})
        # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
        response: ChatCompletion = await self.async_client.chat.completions.create(model=self.model_name,
                                                                                   messages=messages,
                                                                                   temperature=temperature,
                                                                                   max_tokens=max_tokens)
        # generated_content = response['message']['content']
        generated_content = response.choices[0].message.content
        return generated_content
