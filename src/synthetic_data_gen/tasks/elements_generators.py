from abc import abstractmethod, ABC
from asyncio import Queue
from typing import ClassVar, Any

from synthetic_data_gen.llms.llm_client import LLMClientBase
from synthetic_data_gen.tasks.schemas import ClassificationDataGenConfig, GenerationTask, GenerationTaskType, DataGenLabel


class ElementsGenerationProcessorBase(ABC):
    NUMBER_OF_ELEMENTS_PLACEHOLDER: ClassVar[str] = '[NUMBER_OF_ELEMENTS]'

    def __init__(self, llm_client: LLMClientBase, system_message: str | None = None, user_message: str | None = None,
                 temperature: float = 0.6, max_tokens: int = 250):
        self.llm_client = llm_client
        self.system_message = system_message or self._get_system_message()
        self.user_message = user_message or self._get_user_message()
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _get_system_message(self) -> str:
        return f"You are a helpful system that completes the tasks requested by the user."

    @abstractmethod
    def _get_user_message(self) -> str:
        pass

    # def generate_elements(self, number: int, variables: dict[str, Any], temperature: float = 0.6, max_tokens: int = 1000) -> list[str]:
    #     filled_user_message = self.user_message.replace(self.NUMBER_OF_ELEMENTS_PLACEHOLDER, str(number))
    #     filled_user_message = filled_user_message.format_map(variables)
    #     generation_result = self.llm_client.generate(
    #         system_message=self.system_message, user_message=filled_user_message,
    #         temperature=temperature, max_tokens=max_tokens)
    #     generated_elements = self._parse_result(generation_result, number=number)
    #     if len(generated_elements) != number:
    #         raise AssertionError(f'The number of elements is not {number} (it is {len(generated_elements)}):\nGENERATION:\n{generation_result}')
    #     return generated_elements

    async def generate_elements_async(self, number: int, variables: dict[str, Any]) -> list[str]:
        filled_user_message = self.user_message.replace(self.NUMBER_OF_ELEMENTS_PLACEHOLDER, str(number))
        filled_user_message = filled_user_message.format_map(variables)
        generation_result = await self.llm_client.generate_async(
            system_message=self.system_message, user_message=filled_user_message,
            temperature=self.temperature, max_tokens=self.max_tokens)
        generated_elements = self._parse_result(generation_result, number=number)
        if len(generated_elements) != number:
            raise AssertionError(f'The number of elements is not {number} (it is {len(generated_elements)}):\nGENERATION:\n{generation_result}')
        return generated_elements

    def _parse_result(self, result: str, number: int) -> list[str]:
        # The parsing depends on how it is requested to be generated
        # assuming a default one-item-per-line format, to be overridden if not
        return [res.strip() for res in result.strip().split('\n') if res.strip() != ''][:number]

    async def process_generation_task(self,
                                      generation_config: ClassificationDataGenConfig,
                                      generation_task: GenerationTask,
                                      tasks_queue: Queue,
                                      results_queue: Queue):
        generated_elements = await self.call_generate_elements_async(generation_config, generation_task)
        for generated_element in generated_elements:
            await self.extend_generation_task(
                generated_element=generated_element,
                generation_task=generation_task.model_copy(deep=True),
                tasks_queue=tasks_queue,
                results_queue=results_queue
            )
        # notify the task completion
        # note that first of all, in case of recurrent task, a new task has been added for each worker before notifying task done
        # so the queue is never found empty by a worker if there is work to do, and task count never reaches 0 until all work is done
        print(f'Marking task as done...')
        tasks_queue.task_done()

    @abstractmethod
    async def call_generate_elements_async(self, generation_config: ClassificationDataGenConfig, generation_task: GenerationTask):
        pass

    @abstractmethod
    async def extend_generation_task(self, generated_element: str, generation_task: GenerationTask, tasks_queue: Queue, results_queue: Queue):
        pass


class SourceTypesGenerationProcessor(ElementsGenerationProcessorBase):

    def _get_user_message(self) -> str:
        user_message = (f"We are working on generating synthetic data to train a model for a task.\n"
                        f"The task description is:\n\"{{task_description}}\"\n"
                        f"We need to generate examples coming from different possible Internet source types to add variability.\n"
                        f"Please, generate a list of names of possible sources.\n"
                        f"We need {self.NUMBER_OF_ELEMENTS_PLACEHOLDER} different sources.\n"
                        f"Output the list, one element per line.\n"
                        f"IMPORTANT: Do not add anything else, just the elements, your output will be parsed by a subsequent tool and any addition would break it.")
        return user_message

    async def call_generate_elements_async(self, generation_config: ClassificationDataGenConfig, generation_task: GenerationTask) -> list[str]:
        elements_or_number = generation_config.source_types
        if isinstance(elements_or_number, int):
            generated_elements: list[str] = await self.generate_elements_async(
                number=elements_or_number,
                variables={'task_description': generation_config.task_description, **generation_task.model_dump()})
        else:
            generated_elements: list[str] = elements_or_number
        return generated_elements

    async def extend_generation_task(self, generated_element: str, generation_task: GenerationTask, tasks_queue: Queue, results_queue: Queue):
        generation_task.generation_elements_tuple.source_type = generated_element
        generation_task.next_task = GenerationTaskType.TOPICS
        await tasks_queue.put(generation_task)


class TopicsGenerationProcessor(ElementsGenerationProcessorBase):

    def _get_user_message(self) -> str:
        user_message = (f"We are working on generating synthetic data to train a model for a task.\n"
                        f"The task description is:\n\"{{task_description}}\"\n"
                        f"We need to generate examples coming from different realistic topics to add variability.\n"
                        f"Please, generate a list of names of possible topics for examples coming from {{source_type}}.\n"
                        f"We need {self.NUMBER_OF_ELEMENTS_PLACEHOLDER} different topics.\n"
                        f"Output the list, one element per line.\n"
                        f"IMPORTANT: Do not add anything else, just the elements, your output will be parsed by a subsequent tool and any addition would break it.")
        return user_message

    async def call_generate_elements_async(self, generation_config: ClassificationDataGenConfig, generation_task: GenerationTask) -> list[str]:
        elements_or_number = generation_config.topics
        if isinstance(elements_or_number, int):
            generated_elements: list[str] = await self.generate_elements_async(
                number=elements_or_number,
                variables={'task_description': generation_config.task_description, **generation_task.generation_elements_tuple.model_dump()})
        else:
            generated_elements: list[str] = elements_or_number
        return generated_elements

    async def extend_generation_task(self, generated_element: str, generation_task: GenerationTask, tasks_queue: Queue, results_queue: Queue):
        generation_task.generation_elements_tuple.topic = generated_element
        generation_task.next_task = GenerationTaskType.SUBTOPICS
        await tasks_queue.put(generation_task)


class SubtopicsGenerationProcessor(ElementsGenerationProcessorBase):

    def _get_user_message(self) -> str:
        user_message = (f"We are working on generating synthetic data to train a model for a task.\n"
                        f"The task description is:\n\"{{task_description}}\"\n"
                        f"We need to generate examples coming from different realistic subtopics of a given topic and data source to add variability.\n"
                        f"Please, generate a list of names of possible subtopics for the topic {{topic}} coming from {{source_type}}.\n"
                        f"We need just {self.NUMBER_OF_ELEMENTS_PLACEHOLDER} different subtopics.\n"
                        f"Output the list, one element per line.\n"
                        f"IMPORTANT: Do not add anything else, just the elements, your output will be parsed by a subsequent tool and any addition would break it.")
        return user_message

    async def call_generate_elements_async(self, generation_config: ClassificationDataGenConfig, generation_task: GenerationTask) -> list[str]:
        elements_or_number = generation_config.subtopics
        if isinstance(elements_or_number, int):
            generated_elements: list[str] = await self.generate_elements_async(
                number=elements_or_number,
                variables={'task_description': generation_config.task_description, **generation_task.generation_elements_tuple.model_dump()})
        else:
            generated_elements: list[str] = elements_or_number
        return generated_elements

    async def extend_generation_task(self, generated_element: str, generation_task: GenerationTask, tasks_queue: Queue, results_queue: Queue):
        generation_task.generation_elements_tuple.subtopic = generated_element
        generation_task.next_task = GenerationTaskType.PERSONAS
        await tasks_queue.put(generation_task)

class PersonasGenerationProcessor(ElementsGenerationProcessorBase):

    def _get_user_message(self) -> str:
        user_message = (f"We are working on generating synthetic data to train a model for a task.\n"
                        f"The task description is:\n\"{{task_description}}\"\n"
                        f"We need to generate examples coming from different realistic personas to add variability.\n"
                        f"A persona is a brief and concise description of an archetype of a potential user.\n"
                        f"Generate each persona like in the following example: 'Mary is a PERSONA_DESCRIPTION', giving them a realistic name and description.\n"
                        f"Please, generate a list of possible personas that could generate examples for subtopic {{subtopic}} "
                        f"of topic {{topic}} coming from {{source_type}}.\n"
                        f"We need {self.NUMBER_OF_ELEMENTS_PLACEHOLDER} different personas.\n"
                        f"Output the list, one element per line.\n"
                        f"IMPORTANT: Do not add anything else, just the elements, your output will be parsed by a subsequent tool and any addition would break it.")
        return user_message

    async def call_generate_elements_async(self, generation_config: ClassificationDataGenConfig, generation_task: GenerationTask) -> list[str]:
        elements_or_number = generation_config.personas
        if isinstance(elements_or_number, int):
            generated_elements: list[str] = await self.generate_elements_async(
                number=elements_or_number,
                variables={'task_description': generation_config.task_description, **generation_task.generation_elements_tuple.model_dump()})
        else:
            generated_elements: list[str] = elements_or_number
        return generated_elements

    async def extend_generation_task(self, generated_element: str, generation_task: GenerationTask, tasks_queue: Queue, results_queue: Queue):
        generation_task.generation_elements_tuple.persona = generated_element
        generation_task.next_task = GenerationTaskType.LANGUAGES
        await tasks_queue.put(generation_task)

class LanguagesGenerationProcessor(ElementsGenerationProcessorBase):

    def _get_user_message(self) -> str:
        pass

    async def call_generate_elements_async(self, generation_config: ClassificationDataGenConfig, generation_task: GenerationTask) -> list[str]:
        return generation_config.languages

    async def extend_generation_task(self, generated_element: str, generation_task: GenerationTask, tasks_queue: Queue, results_queue: Queue):
        generation_task.generation_elements_tuple.language = generated_element
        generation_task.next_task = GenerationTaskType.LABELS
        await tasks_queue.put(generation_task)


class LabelsGenerationProcessor(ElementsGenerationProcessorBase):

    def _get_user_message(self) -> str:
        pass

    async def call_generate_elements_async(self, generation_config: ClassificationDataGenConfig, generation_task: GenerationTask) -> list[str]:
        return [label.name for label in generation_config.labels]

    async def extend_generation_task(self, generated_element: str, generation_task: GenerationTask, tasks_queue: Queue, results_queue: Queue):
        generation_task.generation_elements_tuple.label = generated_element
        generation_task.next_task = GenerationTaskType.INSTANCES
        await tasks_queue.put(generation_task)


class ClassificationInstancesGenerationProcessor(ElementsGenerationProcessorBase):

    def _get_user_message(self) -> str:
        user_message = (f"We are working on generating synthetic data to train a model for a task.\n"
                        f"The task description is:\n\"{{task_description}}\"\n\n"
                        f"The possible labels are:\n{{labels}}\n\n"
                        f"We need to generate realistic and varied examples.\n"
                        f"Please, generate a list of realistic and varied text messages for label \"{{label}}\", for a subtopic {{subtopic}} "
                        f"of topic {{topic}} coming from {{source_type}}.\n"
                        f"The generated examples must look like as if the had been generated by the following persona:\n"
                        f"\"{{persona}}\"\n"
                        f"We need {self.NUMBER_OF_ELEMENTS_PLACEHOLDER} different examples.\n"
                        f"The examples must be written in {{language}}.\n"
                        f"Output the list, one element per line.\n"
                        f"IMPORTANT: Do not add anything else, just the elements, your output will be parsed by a subsequent tool and any addition would break it.")

        return user_message

    async def call_generate_elements_async(self, generation_config: ClassificationDataGenConfig, generation_task: GenerationTask):
        instances = await self.generate_elements_async(
            number=generation_config.num_instances_per_combination,
            variables={'task_description': generation_config.task_description,
                       'labels': DataGenLabel.pretty_print_list(generation_config.labels),
                       **generation_task.generation_elements_tuple.model_dump()})
        return instances

    async def extend_generation_task(self, generated_element: str, generation_task: GenerationTask, tasks_queue: Queue, results_queue: Queue):
            elements_tuple = generation_task.generation_elements_tuple
            elements_tuple.instance = generated_element
            await results_queue.put(elements_tuple)
