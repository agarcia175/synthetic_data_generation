"""
A first PoC to check participation/need of different elements
"""
from abc import ABC, abstractmethod
from typing import ClassVar, Any

from pydantic import BaseModel
from tqdm import tqdm

from synthetic_data_gen.llms.llm_client import LLMClientBase, OllamaLLMClient


class DataGenLabel(BaseModel):
    name: str
    desc: str


class ClassificationDataGenConfig(BaseModel):
    task_description: str
    labels: list[DataGenLabel]
    languages: list[str]
    source_types: list[str] | int
    topics: list[str] | int
    subtopics: list[str] | int
    personas: list[str] | int
    num_instances_per_combination: int


class VariabilityTuple(BaseModel):
    language: str
    source_type: str
    topic: str
    subtopic: str
    persona: str


class BaseVariabilityElementsGenerator(ABC):
    NUMBER_OF_ELEMENTS_PLACEHOLDER: ClassVar[str] = '[NUMBER_OF_ELEMENTS]'

    def __init__(self, llm_client: LLMClientBase, system_message: str | None = None, user_message: str | None = None):
        self.llm_client = llm_client
        self.system_message = system_message or self._get_system_message()
        self.user_message = user_message or self._get_user_message()

    @abstractmethod
    def _get_system_message(self) -> str:
        pass

    @abstractmethod
    def _get_user_message(self) -> str:
        pass

    def generate_variability_elements(self, number: int, variables: dict[str, Any], temperature: float = 0.6, max_tokens: int = 1000) -> list[str]:
        filled_user_message = self.user_message.replace(self.NUMBER_OF_ELEMENTS_PLACEHOLDER, str(number))
        filled_user_message = filled_user_message.format_map(variables)
        generation_result = self.llm_client.generate(
            system_message=self.system_message, user_message=filled_user_message,
            temperature=temperature, max_tokens=max_tokens)
        variability_elements = self._parse_result(generation_result)
        if len(variability_elements) != number:
            raise AssertionError(f'The number of elements is not {number} (it is {len(variability_elements)}):\nGENERATION:\n{generation_result}')
        return variability_elements

    def _parse_result(self, result: str) -> list[str]:
        # The parsing depends on how it is requested to be generated
        # assuming a default one-item-per-line format, to be overridden if not
        return [res.strip() for res in result.split('\n')]


class SourceTypesGenerator(BaseVariabilityElementsGenerator):

    def __init__(self, llm_client: LLMClientBase):
        super().__init__(llm_client)

    def _get_system_message(self) -> str:
        return f"You are a helpful system that completes the tasks requested by the user."

    def _get_user_message(self) -> str:
        user_message = (f"We are working on generating synthetic data to train a model for a task.\n"
                        f"The task description is:\n\"{{classification_task_description}}\"\n"
                        f"We need to generate examples coming from different possible Internet source types to add variability.\n"
                        f"Please, generate a list of names of possible sources. We need {self.NUMBER_OF_ELEMENTS_PLACEHOLDER} different sources.\n"
                        f"Output the list, one element per line. Do not add anything else, your output will be parsed by a subsequent tool.")
        return user_message


class TopicsGenerator(BaseVariabilityElementsGenerator):

    def __init__(self, llm_client: LLMClientBase):
        super().__init__(llm_client)

    def _get_system_message(self) -> str:
        return f"You are a helpful system that completes the tasks requested by the user."

    def _get_user_message(self) -> str:
        user_message = (f"We are working on generating synthetic data to train a model for a task.\n"
                        f"The task description is:\n\"{{classification_task_description}}\"\n"
                        f"We need to generate examples coming from different realistic topics to add variability.\n"
                        f"Please, generate a list of names of possible topics for examples coming from {{source_type}}.\n"
                        f"We need {self.NUMBER_OF_ELEMENTS_PLACEHOLDER} different topics.\n"
                        f"Output the list, one element per line. Do not add anything else, your output will be parsed by a subsequent tool.")
        return user_message


class SubtopicsGenerator:
    pass


class PersonasGenerator:
    pass


# Note: this is for generation (instance + label), for just labelling, there should be another class
class ClassificationDataGeneration:

    def __init__(self, llm_client: LLMClientBase,
                 generation_config: ClassificationDataGenConfig,
                 source_types_generator: SourceTypesGenerator,
                 topics_generator: TopicsGenerator):
        self._llm_client = llm_client
        self._generation_config = generation_config
        self._source_types_generator = source_types_generator
        self._topics_generator = topics_generator

    def generate_data(self):
        # of course, I am going to try a regular single thread version before going for an asynchronous version
        variability_tuples: list[VariabilityTuple] = []
        if isinstance(self._generation_config.source_types, int):
            source_types: list[str] = self._source_types_generator.generate_variability_elements(
                number=self._generation_config.source_types,
                variables={'classification_task_description': self._generation_config.task_description})
        else:
            source_types = self._generation_config.source_types

        for source_type in tqdm(source_types, desc='Generating topics'):
            topics = self._topics_generator.generate_variability_elements(
                number=self._generation_config.topics,
                variables={'classification_task_description': self._generation_config.task_description,
                           'source_type': source_type})
            for topic in topics:
                variability_tuples.append(VariabilityTuple(language='en', source_type=source_type, topic=topic, subtopic='XXX', persona='XXX'))

        [print(v) for v in variability_tuples]


if __name__ == '__main__':
    config = ClassificationDataGenConfig(
        task_description='A sentiment classification task with positive and negative examples',
        labels=[DataGenLabel(name='positive', desc='A positive piece of content'), DataGenLabel(name='negative', desc='A negative piece of content')],
        languages=['en'],
        source_types=10,
        topics=10,
        subtopics=5,
        personas=5,
        num_instances_per_combination=2,
    )

    llm_client = OllamaLLMClient(model_name='hf.co/unsloth/gemma-3-4b-it-qat-GGUF:Q4_K_M')
    source_types_generator = SourceTypesGenerator(llm_client)
    topics_generator = TopicsGenerator(llm_client)

    ClassificationDataGeneration(llm_client=llm_client, generation_config=config,
                                 source_types_generator=source_types_generator, topics_generator=topics_generator).generate_data()
