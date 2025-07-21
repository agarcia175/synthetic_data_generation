from enum import StrEnum

from pydantic import BaseModel


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
    source_type: str = None
    topic: str = None
    subtopic: str = None
    persona: str = None
    label: str = None
    language: str = None
    instance: str = None


class GenerationTaskType(StrEnum):
    SOURCE_TYPES = 'source_types'
    TOPICS = 'topics'
    SUBTOPICS = 'subtopics'
    PERSONAS = 'personas'
    LABELS = 'labels'
    LANGUAGES = 'languages'
    INSTANCES = 'instances'


class GenerationTask(BaseModel):
    generation_elements_tuple: GenerationElementsTuple
    next_task: GenerationTaskType
