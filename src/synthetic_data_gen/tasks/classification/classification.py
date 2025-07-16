"""
A first PoC to check participation/need of different elements
"""
import asyncio
import time
from asyncio import Queue, Task
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import ClassVar, Any

from pydantic import BaseModel
from tqdm import tqdm

from synthetic_data_gen.llms.llm_client import LLMClientBase, OllamaLLMClient


class DataGenLabel(BaseModel):
    name: str
    desc: str

    @classmethod
    def pretty_print_list(cls, labels: list['DataGenLabel']) -> str:
        lines: list[str] = []
        for label in labels:
            lines.append(f' - {label.name}: {label.desc}')
        return '\n'.join(lines)


class ClassificationDataGenConfig(BaseModel):
    task_description: str
    labels: list[DataGenLabel]
    languages: list[str]
    source_types: list[str] | int
    topics: list[str] | int
    subtopics: list[str] | int
    personas: list[str] | int
    num_instances_per_combination: int


class GenerationElementsTuple(BaseModel):
    language: str = None
    label: str = None
    source_type: str = None
    topic: str = None
    subtopic: str = None
    persona: str = None
    instance: str = None


class GenerationTaskType(StrEnum):
    SOURCE_TYPE = 'source_type'
    TOPIC = 'topic'
    SUBTOPIC = 'subtopic'
    PERSONA = 'persona'
    INSTANCES = 'instances'


class GenerationTask(BaseModel):
    generation_elements_tuple: GenerationElementsTuple
    next_task: GenerationTaskType


class BaseElementsGenerator(ABC):
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

    def generate_elements(self, number: int, variables: dict[str, Any], temperature: float = 0.6, max_tokens: int = 1000) -> list[str]:
        filled_user_message = self.user_message.replace(self.NUMBER_OF_ELEMENTS_PLACEHOLDER, str(number))
        filled_user_message = filled_user_message.format_map(variables)
        generation_result = self.llm_client.generate(
            system_message=self.system_message, user_message=filled_user_message,
            temperature=temperature, max_tokens=max_tokens)
        generated_elements = self._parse_result(generation_result, number=number)
        if len(generated_elements) != number:
            raise AssertionError(f'The number of elements is not {number} (it is {len(generated_elements)}):\nGENERATION:\n{generation_result}')
        return generated_elements

    async def generate_elements_async(self, number: int, variables: dict[str, Any], temperature: float = 0.6, max_tokens: int = 1000) -> list[str]:
        filled_user_message = self.user_message.replace(self.NUMBER_OF_ELEMENTS_PLACEHOLDER, str(number))
        filled_user_message = filled_user_message.format_map(variables)
        generation_result = await self.llm_client.generate_async(
            system_message=self.system_message, user_message=filled_user_message,
            temperature=temperature, max_tokens=max_tokens)
        generated_elements = self._parse_result(generation_result, number=number)
        if len(generated_elements) != number:
            raise AssertionError(f'The number of elements is not {number} (it is {len(generated_elements)}):\nGENERATION:\n{generation_result}')
        return generated_elements

    def _parse_result(self, result: str, number: int) -> list[str]:
        # The parsing depends on how it is requested to be generated
        # assuming a default one-item-per-line format, to be overridden if not
        return [res.strip() for res in result.strip().split('\n') if res.strip() != ''][:number]


class SourceTypesGenerator(BaseElementsGenerator):

    def __init__(self, llm_client: LLMClientBase):
        super().__init__(llm_client)

    def _get_system_message(self) -> str:
        return f"You are a helpful system that completes the tasks requested by the user."

    def _get_user_message(self) -> str:
        user_message = (f"We are working on generating synthetic data to train a model for a task.\n"
                        f"The task description is:\n\"{{task_description}}\"\n"
                        f"We need to generate examples coming from different possible Internet source types to add variability.\n"
                        f"Please, generate a list of names of possible sources.\n"
                        f"We need {self.NUMBER_OF_ELEMENTS_PLACEHOLDER} different sources.\n"
                        f"Output the list, one element per line.\n"
                        f"IMPORTANT: Do not add anything else, just the elements, your output will be parsed by a subsequent tool and any addition would break it.")
        return user_message


class TopicsGenerator(BaseElementsGenerator):

    def __init__(self, llm_client: LLMClientBase):
        super().__init__(llm_client)

    def _get_system_message(self) -> str:
        return f"You are a helpful system that completes the tasks requested by the user."

    def _get_user_message(self) -> str:
        user_message = (f"We are working on generating synthetic data to train a model for a task.\n"
                        f"The task description is:\n\"{{task_description}}\"\n"
                        f"We need to generate examples coming from different realistic topics to add variability.\n"
                        f"Please, generate a list of names of possible topics for examples coming from {{source_type}}.\n"
                        f"We need {self.NUMBER_OF_ELEMENTS_PLACEHOLDER} different topics.\n"
                        f"Output the list, one element per line.\n"
                        f"IMPORTANT: Do not add anything else, just the elements, your output will be parsed by a subsequent tool and any addition would break it.")
        return user_message


class SubtopicsGenerator(BaseElementsGenerator):

    def __init__(self, llm_client: LLMClientBase):
        super().__init__(llm_client)

    def _get_system_message(self) -> str:
        return f"You are a helpful system that completes the tasks requested by the user."

    def _get_user_message(self) -> str:
        user_message = (f"We are working on generating synthetic data to train a model for a task.\n"
                        f"The task description is:\n\"{{task_description}}\"\n"
                        f"We need to generate examples coming from different realistic subtopics of a given topic and data source to add variability.\n"
                        f"Please, generate a list of names of possible subtopics for the topic {{topic}} coming from {{source_type}}.\n"
                        f"We need just {self.NUMBER_OF_ELEMENTS_PLACEHOLDER} different subtopics.\n"
                        f"Output the list, one element per line.\n"
                        f"IMPORTANT: Do not add anything else, just the elements, your output will be parsed by a subsequent tool and any addition would break it.")
        return user_message


class PersonasGenerator(BaseElementsGenerator):

    def __init__(self, llm_client: LLMClientBase):
        super().__init__(llm_client)

    def _get_system_message(self) -> str:
        return f"You are a helpful system that completes the tasks requested by the user."

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


class InstancesGenerator(BaseElementsGenerator):
    def __init__(self, llm_client: LLMClientBase):
        super().__init__(llm_client)

    def _get_system_message(self) -> str:
        return f"You are a helpful system that completes the tasks requested by the user."

    def _get_user_message(self) -> str:
        user_message = (f"We are working on generating synthetic data to train a model for a task.\n"
                        f"The task description is:\n\"{{task_description}}\"\n\n"
                        f"The possible labels are:\n{{labels}}\n\n"
                        f"We need to generate realistic and varied examples.\n"
                        f"Please, generate a list of realistic and varied text contents for label \"{{label}}\", for a subtopic {{subtopic}} "
                        f"of topic {{topic}} coming from {{source_type}}.\n"
                        f"The generated examples must look like as if the had been generated by the following persona:\n"
                        f"\"{{persona}}\"\n"
                        f"We need {self.NUMBER_OF_ELEMENTS_PLACEHOLDER} different examples.\n"
                        f"Output the list, one element per line.\n"
                        f"IMPORTANT: Do not add anything else, just the elements, your output will be parsed by a subsequent tool and any addition would break it.")

        return user_message


# Note: this is for generation (instance + label), for just labelling, there should be another class
class ClassificationDataGeneration:

    def __init__(self, llm_client: LLMClientBase,
                 generation_config: ClassificationDataGenConfig,
                 source_types_generator: SourceTypesGenerator,
                 topics_generator: TopicsGenerator,
                 subtopics_generator: SubtopicsGenerator,
                 personas_generator: PersonasGenerator,
                 instances_generator: InstancesGenerator):
        self._llm_client = llm_client
        self._generation_config = generation_config
        self._source_types_generator = source_types_generator
        self._topics_generator = topics_generator
        self._subtopics_generator = subtopics_generator
        self._personas_generator = personas_generator
        self._instances_generator = instances_generator
        ###########

    def generate_data(self):
        # of course, I am going to try a regular single thread version before going for an asynchronous version
        generation_elements_tuples: list[GenerationElementsTuple] = []
        if isinstance(self._generation_config.source_types, int):
            source_types: list[str] = self._source_types_generator.generate_elements(
                number=self._generation_config.source_types,
                variables={'task_description': self._generation_config.task_description})
        else:
            source_types = self._generation_config.source_types

        for source_type in tqdm(source_types, desc='Generating topics'):
            topics = self._topics_generator.generate_elements(
                number=self._generation_config.topics,
                variables={'task_description': self._generation_config.task_description,
                           'source_type': source_type})
            for topic in tqdm(topics, desc='Generating subtopics'):
                subtopics = self._subtopics_generator.generate_elements(
                    number=self._generation_config.subtopics,
                    variables={'task_description': self._generation_config.task_description,
                               'source_type': source_type, 'topic': topic})
                for subtopic in tqdm(subtopics, desc='Generating personas'):
                    personas = self._personas_generator.generate_elements(
                        number=self._generation_config.personas,
                        variables={'task_description': self._generation_config.task_description,
                                   'source_type': source_type, 'topic': topic, 'subtopic': subtopic}
                    )
                    for persona in personas:
                        generation_elements_tuples.append(
                            GenerationElementsTuple(language='en', source_type=source_type, topic=topic, subtopic=subtopic, persona=persona,
                                                    label='xx', instance='xx'))

        [print(v) for v in generation_elements_tuples]
        print(len(generation_elements_tuples))

    async def generate_data_async(self, num_workers: int = 3):
        worker_coroutines:list[Task] = []
        tasks_queue = Queue()
        results_queue = Queue()
        for i in range(num_workers):
            worker_coroutine = asyncio.create_task(self._worker_logic(i, tasks_queue, results_queue))
            worker_coroutines.append(worker_coroutine)

        tasks = [GenerationTask(generation_elements_tuple=GenerationElementsTuple(), next_task=GenerationTaskType.SOURCE_TYPE)]

        for task_data in tasks:
            await tasks_queue.put(task_data)

        await tasks_queue.join()

        for _ in worker_coroutines:
            await tasks_queue.put(None)

        resulting_instances: list[GenerationElementsTuple] = []
        while not results_queue.empty():
            resulting_instances.append(await results_queue.get())

        await asyncio.gather(*worker_coroutines)  # Wait for workers to gracefully exit
        print("All workers shut down.")
        return resulting_instances

    async def _worker_logic(self, worker_id: int, tasks_queue: asyncio.Queue[GenerationTask], results_queue: asyncio.Queue[GenerationElementsTuple]):
        while True:
            try:
                print(f'Worker {worker_id} is waiting for task...')
                generation_task = await tasks_queue.get()
                if generation_task is None:  # Fallback sentinel, good for clarity but cancellation is primary
                    tasks_queue.task_done()
                    print(f"Worker {worker_id}: Received sentinel, exiting.")
                    break
                else:
                    print(f'Worker {worker_id} got a task: {generation_task}')
                generation_elements_tuple = generation_task.generation_elements_tuple
                if generation_task.next_task == GenerationTaskType.SOURCE_TYPE:
                    source_types: list[str] = await self._source_types_generator.generate_elements_async(
                        number=self._generation_config.source_types,
                        variables={'task_description': self._generation_config.task_description})
                    for source_type in source_types:
                        extended_generation_task = generation_task.model_copy(deep=True)
                        extended_generation_task.generation_elements_tuple.source_type = source_type
                        extended_generation_task.next_task = GenerationTaskType.TOPIC
                        await tasks_queue.put(extended_generation_task)
                elif generation_task.next_task == GenerationTaskType.TOPIC:
                    topics = await self._topics_generator.generate_elements_async(
                        number=self._generation_config.topics,
                        variables={'task_description': self._generation_config.task_description,
                                   'source_type': generation_elements_tuple.source_type})
                    for topic in topics:
                        extended_generation_task = generation_task.model_copy(deep=True)
                        extended_generation_task.generation_elements_tuple.topic = topic
                        extended_generation_task.next_task = GenerationTaskType.SUBTOPIC
                        await tasks_queue.put(extended_generation_task)
                elif generation_task.next_task == GenerationTaskType.SUBTOPIC:
                    subtopics = await self._subtopics_generator.generate_elements_async(
                        number=self._generation_config.subtopics,
                        variables={'task_description': self._generation_config.task_description,
                                   'source_type': generation_elements_tuple.source_type, 'topic': generation_elements_tuple.topic})
                    for subtopic in subtopics:
                        extended_generation_task = generation_task.model_copy(deep=True)
                        extended_generation_task.generation_elements_tuple.subtopic = subtopic
                        extended_generation_task.next_task = GenerationTaskType.PERSONA
                        await tasks_queue.put(extended_generation_task)
                elif generation_task.next_task == GenerationTaskType.PERSONA:
                    personas = await self._personas_generator.generate_elements_async(
                        number=self._generation_config.personas,
                        variables={'task_description': self._generation_config.task_description,
                                   'source_type': generation_elements_tuple.source_type,
                                   'topic': generation_elements_tuple.topic,
                                   'subtopic': generation_elements_tuple.subtopic}
                    )
                    for persona in personas:
                        # labels (and languages!) are iterated here, at the last "variability element" step, since the rest of the elements
                        for language in self._generation_config.languages:
                            for label in self._generation_config.labels:
                                extended_generation_task = generation_task.model_copy(deep=True)
                                extended_generation_task.generation_elements_tuple.persona = persona
                                extended_generation_task.generation_elements_tuple.language = language
                                extended_generation_task.generation_elements_tuple.label = label.name  # NOTE: is name enough, having the list of labels in the prompt?
                                extended_generation_task.next_task = GenerationTaskType.INSTANCES
                                await tasks_queue.put(extended_generation_task)

                elif generation_task.next_task == GenerationTaskType.INSTANCES:
                    instances = await self._instances_generator.generate_elements_async(
                        number=self._generation_config.num_instances_per_combination,
                        variables={'task_description': self._generation_config.task_description,
                                   'labels': DataGenLabel.pretty_print_list(self._generation_config.labels),
                                   **generation_elements_tuple.model_dump()})
                    for instance in instances:
                        extended_elements_tuple = generation_elements_tuple.model_copy()
                        extended_elements_tuple.instance = instance
                        await results_queue.put(extended_elements_tuple)
                else:
                    raise Exception(f"Unknown task: {generation_task.next_task}")
                # notify the task completion
                # note that first of all, in case of recurrent task, a new task has been added for each worker before notifying task done
                # so the queue is never found empty by a worker if there is work to do, and task count never reaches 0 until all work is done
                print(f'Marking task as done...')
                tasks_queue.task_done()
            except Exception as e:
                # exceptions are "stored" in the task, they do not immediately pop up, possibly blocking the program (as it happened...)
                tasks_queue.task_done()
                print(f'Exception: {e}')
                raise e



if __name__ == '__main__':
    config = ClassificationDataGenConfig(
        task_description='A sentiment classification task with positive and negative examples',
        labels=[DataGenLabel(name='positive', desc='A positive piece of content'), DataGenLabel(name='negative', desc='A negative piece of content')],
        languages=['en'],
        source_types=2,
        topics=2,
        subtopics=2,
        personas=2,
        num_instances_per_combination=2,
    )

    llm_client = OllamaLLMClient(model_name='hf.co/unsloth/gemma-3-4b-it-qat-GGUF:Q4_K_M')
    source_types_generator = SourceTypesGenerator(llm_client)
    topics_generator = TopicsGenerator(llm_client)
    subtopics_generator = SubtopicsGenerator(llm_client)
    personas_generator = PersonasGenerator(llm_client)
    instances_generator = InstancesGenerator(llm_client)

    generator = ClassificationDataGeneration(llm_client=llm_client, generation_config=config,
                                             source_types_generator=source_types_generator,
                                             topics_generator=topics_generator,
                                             subtopics_generator=subtopics_generator,
                                             personas_generator=personas_generator,
                                             instances_generator=instances_generator,
                                             )

    start_time = time.time()
    results = asyncio.run(generator.generate_data_async(num_workers=3))
    end_time = time.time()

    for result in results:
        print(result)


    print(f'Num results: {len(results)}')
    print(f'Time elapsed: {end_time - start_time:.2f} seconds')
