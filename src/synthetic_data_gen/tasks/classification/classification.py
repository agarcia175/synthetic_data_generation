"""
A first PoC to check participation/need of different elements
"""
import asyncio
import time
from asyncio import Queue, Task

from synthetic_data_gen.llms.llm_client import LLMClientBase, OllamaLLMClient
from synthetic_data_gen.tasks.elements_generators import SourceTypesGenerationProcessor, TopicsGenerationProcessor, SubtopicsGenerationProcessor, \
    PersonasGenerationProcessor, \
    ClassificationInstancesGenerationProcessor, ElementsGenerationProcessorBase, LanguagesGenerationProcessor, LabelsGenerationProcessor
from synthetic_data_gen.tasks.schemas import DataGenLabel, ClassificationDataGenConfig, GenerationElementsTuple, GenerationTask, GenerationTaskType


# Note: this is for generation (instance + label), for just labelling, there should be another class
class ClassificationDataGeneration:

    def __init__(self, llm_client: LLMClientBase,
                 generation_config: ClassificationDataGenConfig,
                 generation_processors_map: dict[GenerationTaskType, ElementsGenerationProcessorBase],
                 # source_types_generator: SourceTypesGenerationProcessor,
                 # topics_generator: TopicsGenerationProcessor,
                 # subtopics_generator: SubtopicsGenerationProcessor,
                 # personas_generator: PersonasGenerationProcessor,
                 # instances_generator: ClassificationInstancesGenerationProcessor
                 ):
        self._llm_client = llm_client
        self._generation_config = generation_config
        # self._source_types_generator = source_types_generator
        # self._topics_generator = topics_generator
        # self._subtopics_generator = subtopics_generator
        # self._personas_generator = personas_generator
        # self._instances_generator = instances_generator
        self._generation_processors_map = generation_processors_map
        ###########

    async def generate_data_async(self, num_workers: int = 3):
        worker_coroutines: list[Task] = []
        tasks_queue = Queue()
        results_queue = Queue()
        for i in range(num_workers):
            worker_coroutine = asyncio.create_task(self._worker_logic(i, tasks_queue, results_queue))
            worker_coroutines.append(worker_coroutine)

        tasks = [GenerationTask(generation_elements_tuple=GenerationElementsTuple(), next_task=GenerationTaskType.SOURCE_TYPES)]

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

    async def _worker_logic(self, worker_id: int, tasks_queue: Queue[GenerationTask], results_queue: Queue[GenerationElementsTuple]):
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
                if generation_task.next_task in self._generation_processors_map:
                    await self._generation_processors_map[generation_task.next_task].process_generation_task(
                        generation_config=self._generation_config,
                        generation_task=generation_task,
                        tasks_queue=tasks_queue,
                        results_queue=results_queue,
                    )
                else:
                    raise Exception(f"Unknown task: {generation_task.next_task}")

            except Exception as e:
                # exceptions are "stored" in the task, they do not immediately pop up, possibly blocking the program (as it happened...)
                tasks_queue.task_done()
                print(f'Exception: {e}')
                raise e


if __name__ == '__main__':
    config = ClassificationDataGenConfig(
        task_description='A sentiment classification task with positive and negative examples',
        labels=[DataGenLabel(name='positive', desc='A positive piece of content'), DataGenLabel(name='negative', desc='A negative piece of content')],
        languages=['Spanish'],
        source_types=2,
        topics=2,
        subtopics=2,
        personas=2,
        num_instances_per_combination=2,
    )

    llm_client = OllamaLLMClient(model_name='hf.co/unsloth/gemma-3-4b-it-qat-GGUF:Q4_K_M')
    source_types_generator = SourceTypesGenerationProcessor(llm_client)
    topics_generator = TopicsGenerationProcessor(llm_client)
    subtopics_generator = SubtopicsGenerationProcessor(llm_client)
    personas_generator = PersonasGenerationProcessor(llm_client)
    languages_generator = LanguagesGenerationProcessor(llm_client)
    labels_generator = LabelsGenerationProcessor(llm_client)
    instances_generator = ClassificationInstancesGenerationProcessor(llm_client, temperature=0.6, max_tokens=768)

    generation_processors_map: dict[GenerationTaskType, ElementsGenerationProcessorBase] = {
        GenerationTaskType.SOURCE_TYPES: source_types_generator,
        GenerationTaskType.TOPICS: topics_generator,
        GenerationTaskType.SUBTOPICS: subtopics_generator,
        GenerationTaskType.PERSONAS: personas_generator,
        GenerationTaskType.LANGUAGES: languages_generator,
        GenerationTaskType.LABELS: labels_generator,
        GenerationTaskType.INSTANCES: instances_generator,
    }

    generator = ClassificationDataGeneration(
        llm_client=llm_client,
        generation_config=config,
        generation_processors_map=generation_processors_map
    )

    start_time = time.time()
    results = asyncio.run(generator.generate_data_async(num_workers=3))
    end_time = time.time()

    for result in results:
        print(result)

    print(f'Num results: {len(results)}')
    print(f'Time elapsed: {end_time - start_time:.2f} seconds')
