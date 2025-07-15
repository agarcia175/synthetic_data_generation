## Synthetic data generator

Based on LLMs, to generate (or label) synthetic data for different type of tasks (extensible).

The main elements are:
  - the variability components that create the variation during the generation
    - their prompts (templates)
  - the task description
  - the task outcome format
  - the full prompt assembler
  - the main component to run all


The main flow of the process is the following.

 - There is a main description of the task, e.g. "You are a system that generates..."
 - There is a description of the input/output format (derived from Pydantic models).
 - Maybe few-shots? (derived from manual instances of Pydantic models)
 - For each variability component:
   - there is a description of the component (plus the instance of the component itself)
   - components might be generated or passed manually (they need their prompt and config, e.g. amount of them)

Then:
 - The full prompt is composed out of it.
 - An LLM is called for generation, iteratively.
 - The output is parsed into the expected format
 - The instances are stacked and stored


### Potential types of tasks

 - Classification:
   - the usual classification problem for a single label
   - maybe also for multi-label? (including valid label combinations for each problem?)
   - what about regression? (it would be a special case of classification with literal labels covering a 
   gradation, a range, and them assign a float to the label: e.g: 5-stars into 0.0/0.25/0.5/0.75/1.0)

 - Sequence-labelling:
   - for NER and similar tasks, selecting a sequence of words and assigning them a label (or labels)

 - Question-answering? (how would it be?)
   - RAG related? (blobs of text, and pairs of Q&A from it?)
     - Would require validation to ensure quality, diversity and correction

Honestly, these cover most (if not all) of the potentially interesting tasks.

You may think into even more complex stuff such as multitask. But that would boil down to first generate
data for one task, and the use the generated examples to label them automatically, and combining labels.