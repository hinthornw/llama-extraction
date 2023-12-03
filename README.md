# Prompt Engineering Extraction Experiments

This benchmark is interesting for a couple of reasons. It's relatively straightforward to saturate by building a model for each field and ensembling, but to have the model output it
all in one go is actually a challenge since it involves:

1. Following a strict nested JSON schema.
2. Understanding the meaning of the nesting and/or the instructions for the API.
3. Being able to classify across many classes.

OpenAI's function calling endpoints are very good at this. Anthropic's Claude-2 models are OK at this. Many of the other Llama-based models are not excellent at this off-the bat,
and make decent trade-offs to get there. 

This notebook compares 3 open models and then takes a single Llama-v2 based model (`llama-v2-34b-code-instruct`, in particular) and applies various prompting strategies to see if they improve performance.


```python
# %pip install -U --quiet langchain langchain_benchmarks
# %pip install -U openai rapidfuzz fireworks-ai anthropic pandas replicate
```

For this code to work, please configure LangSmith environment variables with your credentials,
in addition to your LLM providers' API keys.


```python
import getpass
import os
import uuid

uid = uuid.uuid4().hex[:4]  # Avoid conflicts in project names

# Get your API key from https://smith.langchain.com/settings
api_keys = [
    "LANGCHAIN_API_KEY",
    "FIREWORKS_API_KEY",
    "REPLICATE_API_TOKEN"
]
for key in api_keys:
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"Enter your {key}: ")
```


```python
from langchain_benchmarks import clone_public_dataset, registry

task = registry["Chat Extraction"]

# Clone the dataset to your tenant
clone_public_dataset(task.dataset_id, dataset_name=task.name)

task
```

    Dataset Chat Extraction already exists. Skipping.
    You can access the dataset at https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/08042749-504d-4509-9549-5f5c579115f6.





<table>
<tbody>
<tr><td>Name       </td><td>Chat Extraction                                                                                                                                            </td></tr>
<tr><td>Type       </td><td>ExtractionTask                                                                                                                                             </td></tr>
<tr><td>Dataset ID </td><td><a href="https://smith.langchain.com/public/00f4444c-9460-4a82-b87a-f50096f1cfef/d" target="_blank" rel="noopener">00f4444c-9460-4a82-b87a-f50096f1cfef</a></td></tr>
<tr><td>Description</td><td>A dataset meant to test the ability of an LLM to extract and infer
structured information from a dialogue. The dialogue is between a user and a support
engineer. Outputs should be structured as a JSON object and test both the ability
of the LLM to correctly structure the information and its ability to perform simple 
classification tasks.                                                                                                                                                            </td></tr>
</tbody>
</table>



#### Schema

Each extraction task has an expected output schema defined in a Pydantic BaseModel object, which we can use to
get a JSON schema object.


```python
task.schema.schema()
```




    {'title': 'GenerateTicket',
     'description': 'Generate a ticket containing all the extracted information.',
     'type': 'object',
     'properties': {'issue_summary': {'title': 'Issue Summary',
       'description': 'short (<10 word) summary of the issue or question',
       'type': 'string'},
      'question': {'title': 'Question',
       'description': 'Information inferred from the the question.',
       'allOf': [{'$ref': '#/definitions/QuestionCategorization'}]},
      'response': {'title': 'Response',
       'description': 'Information inferred from the the response.',
       'allOf': [{'$ref': '#/definitions/ResponseCategorization'}]}},
     'required': ['issue_summary', 'question', 'response'],
     'definitions': {'QuestionCategory': {'title': 'QuestionCategory',
       'description': 'An enumeration.',
       'enum': ['Implementation Issues',
        'Feature Requests',
        'Concept Explanations',
        'Code Optimization',
        'Security and Privacy Concerns',
        'Model Training and Fine-tuning',
        'Data Handling and Manipulation',
        'User Interaction Flow',
        'Technical Integration',
        'Error Handling and Logging',
        'Customization and Configuration',
        'External API and Data Source Integration',
        'Language and Localization',
        'Streaming and Real-time Processing',
        'Tool Development',
        'Function Calling',
        'LLM Integrations',
        'General Agent Question',
        'General Chit Chat',
        'Memory',
        'Debugging Help',
        'Application Design',
        'Prompt Templates',
        'Cost Tracking',
        'Other'],
       'type': 'string'},
      'Sentiment': {'title': 'Sentiment',
       'description': 'An enumeration.',
       'enum': ['Negative', 'Neutral', 'Positive'],
       'type': 'string'},
      'ProgrammingLanguage': {'title': 'ProgrammingLanguage',
       'description': 'An enumeration.',
       'enum': ['python', 'javascript', 'typescript', 'unknown', 'other'],
       'type': 'string'},
      'QuestionCategorization': {'title': 'QuestionCategorization',
       'type': 'object',
       'properties': {'question_category': {'$ref': '#/definitions/QuestionCategory'},
        'category_if_other': {'title': 'Category If Other',
         'description': "question category if the category above is 'other'",
         'type': 'string'},
        'is_off_topic': {'title': 'Is Off Topic',
         'description': 'If the input is general chit chat or does not pertain to technical inqueries about LangChain or building/debugging applications with LLMs/AI, it is off topic. For context, LangChain is a library and framework designed to assist in building applications with LLMs. Questions may also be about similar packages like LangServe, LangSmith, OpenAI, Anthropic, vectorstores, agents, etc.',
         'type': 'boolean'},
        'toxicity': {'title': 'Toxicity',
         'description': 'Whether or not the input question is toxic',
         'default': 0,
         'exclusiveMaximum': 6,
         'minimum': 0,
         'type': 'integer'},
        'sentiment': {'$ref': '#/definitions/Sentiment'},
        'programming_language': {'$ref': '#/definitions/ProgrammingLanguage'}},
       'required': ['question_category',
        'is_off_topic',
        'sentiment',
        'programming_language']},
      'ResponseType': {'title': 'ResponseType',
       'description': 'An enumeration.',
       'enum': ['resolve issue',
        'provide guidance',
        'request information',
        'give up',
        'none',
        'other'],
       'type': 'string'},
      'ResponseCategorization': {'title': 'ResponseCategorization',
       'type': 'object',
       'properties': {'response_type': {'$ref': '#/definitions/ResponseType'},
        'response_type_if_other': {'title': 'Response Type If Other',
         'type': 'string'},
        'confidence_level': {'title': 'Confidence Level',
         'description': 'The confidence of the assistant in its answer.',
         'exclusiveMaximum': 6,
         'minimum': 0,
         'type': 'integer'},
        'followup_actions': {'title': 'Followup Actions',
         'description': 'Actions the assistant recommended the user take.',
         'type': 'array',
         'items': {'type': 'string'}}},
       'required': ['response_type', 'confidence_level']}}}



Now it's time to measure our chain's effectiveness!

## Baseline

We will experiment with three fairly large open-source model LLMs to see their baseline performance.


```python
import json
from typing import Dict, Callable, Type, Any, Sequence, Union, Optional
from langchain_core.messages import BaseMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatFireworks
from langchain.output_parsers.json import parse_json_markdown
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import LanguageModelInput
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator

llama_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a data extraction bot tasked with extracting and inferring information from dialogues and generating tickets. Always respond "
            "only with json based on the following JSON schema:\n\n{schema}",
        ),
        (
            "user",
            "Generate a ticket from the following question-response pair:\n"
            "<Dialogue>\n{dialogue}\n</Dialogue>\n"
            "Remember, respond directly with this format:\n"
            '{{"{function_call}": ...}}\n'
            "RESPOND ONLY IN JSON",
        ),
    ]
)

prompt = llama_prompt.partial(
    schema=task.schema.schema_json(), function_call=task.schema.schema()["title"]
)

llama_llm = ChatFireworks(
    model="accounts/fireworks/models/llama-v2-34b-code-instruct",
    model_kwargs={"max_tokens": 4000, "temperature": 0},
)


def format_run(dialogue_input: dict):
    question = dialogue_input["question"]
    answer = dialogue_input["answer"]
    return {
        "dialogue": f"<question>\n{question}\n</question>\n"
        f"<assistant-response>\n{answer}\n</assistant-response>"
    }


def parse_output(ai_message):
    content = ai_message.content.strip()
    if content.endswith('</s>'):
        content = content.replace('</s>', '')
    parser = lambda x: json.loads(x, strict=False)
    try:
        parsed = parse_json_markdown(content, parser=parser)
        if "GenerateTicket" in parsed:
            return {"output": parsed["GenerateTicket"]}
        return {"output": parsed}
    except json.JSONDecodeError:
        return {"output": content}


def create_extraction_chain(prompt, llm=llama_llm):
    return format_run | prompt | llm | parse_output


fireworks_extraction_chain = create_extraction_chain(prompt)
fireworks_extraction_chain.invoke(
    {
        "question": "what's the square root of 3.14?",
        "answer": "not my business.",
    }
)
```




    {'output': {'issue_summary': 'Square root of 3.14',
      'question': {'question_category': 'Mathematics',
       'is_off_topic': False,
       'toxicity': 0,
       'sentiment': 'Neutral',
       'programming_language': 'unknown'},
      'response': {'response_type': 'give up',
       'confidence_level': 5,
       'followup_actions': []}}}




```python
from langsmith.client import Client

from langchain_benchmarks.extraction.tasks.chat_extraction import get_eval_config

client = Client()

eval_config = get_eval_config()
```


```python
models_to_try = [
    "llama-v2-34b-code-instruct",
    "llama-v2-70b-chat",
    "yi-34b-200k-capybara",
]


fireworks_extraction_chain.invoke(
    {
        "question": "what's the square root of 3.14?",
        "answer": "not my business.",
    }
)

test_runs = {}
for model_name in models_to_try:
    llm = ChatFireworks(
        model=f"accounts/fireworks/models/{model_name}",
        model_kwargs={"max_tokens": 4000, "temperature": 0},
    )
    fireworks_extraction_chain = create_extraction_chain(prompt, llm=llm)
    test_runs[model_name] = client.run_on_dataset(
        dataset_name=task.name,
        llm_or_chain_factory=fireworks_extraction_chain,
        evaluation=eval_config,
        verbose=True,
        project_name=f"{model_name}-{uuid.uuid4().hex[:4]}-v1",
        project_metadata={"arch": "base", "model": model_name},
    )
```

    View the evaluation results for project 'llama-v2-34b-code-instruct-bcce-v1' at:
    https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/08042749-504d-4509-9549-5f5c579115f6/compare?selectedSessions=2bb03041-c9e5-4040-bf4f-81b19e5c2ad4
    
    View all tests for Dataset Chat Extraction at:
    https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/08042749-504d-4509-9549-5f5c579115f6
    [------------------------------------------------->] 27/27


<h3>Experiment Results:</h3>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feedback.json_edit_distance</th>
      <th>feedback.json_schema</th>
      <th>feedback.toxicity_similarity</th>
      <th>feedback.sentiment_similarity</th>
      <th>feedback.confidence_level_similarity</th>
      <th>feedback.question_category</th>
      <th>feedback.off_topic_similarity</th>
      <th>feedback.programming_language_similarity</th>
      <th>error</th>
      <th>execution_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.0</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>0</td>
      <td>27.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.412327</td>
      <td>0.888889</td>
      <td>1.0</td>
      <td>0.592593</td>
      <td>0.933333</td>
      <td>0.074074</td>
      <td>0.888889</td>
      <td>0.444444</td>
      <td>NaN</td>
      <td>4.296252</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.158415</td>
      <td>0.320256</td>
      <td>0.0</td>
      <td>0.197924</td>
      <td>0.200000</td>
      <td>0.266880</td>
      <td>0.320256</td>
      <td>0.506370</td>
      <td>NaN</td>
      <td>0.827581</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.094092</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>3.434230</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.308532</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.500000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>3.777280</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.387863</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.500000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>4.140211</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.514832</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.500000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>4.427954</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.726651</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>7.327288</td>
    </tr>
  </tbody>
</table>
</div>


    View the evaluation results for project 'llama-v2-70b-chat-28a7-v1' at:
    https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/08042749-504d-4509-9549-5f5c579115f6/compare?selectedSessions=cd595a34-012c-4df2-849e-a7e2908d4c81
    
    View all tests for Dataset Chat Extraction at:
    https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/08042749-504d-4509-9549-5f5c579115f6
    [------------------------------------------------->] 27/27


<h3>Experiment Results:</h3>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feedback.json_edit_distance</th>
      <th>feedback.json_schema</th>
      <th>feedback.toxicity_similarity</th>
      <th>feedback.sentiment_similarity</th>
      <th>feedback.confidence_level_similarity</th>
      <th>feedback.question_category</th>
      <th>feedback.off_topic_similarity</th>
      <th>feedback.programming_language_similarity</th>
      <th>error</th>
      <th>execution_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>18.000000</td>
      <td>27.000000</td>
      <td>27.0</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>0</td>
      <td>27.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.639051</td>
      <td>0.037037</td>
      <td>0.0</td>
      <td>0.296296</td>
      <td>0.296296</td>
      <td>0.037037</td>
      <td>0.296296</td>
      <td>0.148148</td>
      <td>NaN</td>
      <td>6.222369</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.148511</td>
      <td>0.192450</td>
      <td>0.0</td>
      <td>0.465322</td>
      <td>0.405236</td>
      <td>0.192450</td>
      <td>0.465322</td>
      <td>0.362014</td>
      <td>NaN</td>
      <td>3.403553</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.370968</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>3.510102</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.543340</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>4.452219</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.648100</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>5.013310</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.744218</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.800000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>7.098748</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.924549</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>19.787303</td>
    </tr>
  </tbody>
</table>
</div>


    View the evaluation results for project 'yi-34b-200k-capybara-9ac9-v1' at:
    https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/08042749-504d-4509-9549-5f5c579115f6/compare?selectedSessions=07ea7e7c-3743-4b51-af75-9194dc8a7205
    
    View all tests for Dataset Chat Extraction at:
    https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/08042749-504d-4509-9549-5f5c579115f6
    [------------------------------------------------->] 27/27


<h3>Experiment Results:</h3>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feedback.json_edit_distance</th>
      <th>feedback.json_schema</th>
      <th>feedback.toxicity_similarity</th>
      <th>feedback.sentiment_similarity</th>
      <th>feedback.confidence_level_similarity</th>
      <th>feedback.question_category</th>
      <th>feedback.off_topic_similarity</th>
      <th>feedback.programming_language_similarity</th>
      <th>error</th>
      <th>execution_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>0</td>
      <td>27.0</td>
      <td>27.0</td>
      <td>27.0</td>
      <td>27.0</td>
      <td>27.0</td>
      <td>27.0</td>
      <td>27.0</td>
      <td>0</td>
      <td>27.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3.927253</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.409956</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3.279934</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3.616294</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3.823253</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>4.138749</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>4.988713</td>
    </tr>
  </tbody>
</table>
</div>


#### Results

Reviewing the tables above, we can see that the naive applications of Llama-v2-70B-chat
and Yi-34B aren't generating json schema in the way we'd like.

Even for the 'best performing' model, `llama-v2-34b-code-instruct`, there are some fairly big problems here:

1. Json schema isn't always honored.
2. Sentiment prediction is surprisingly poor
3. Programming language similarity is bad.
4. Question category is bad (this is less surprising)


Let's review some of the specific outputs to see more.


```python
llama_v2_test_run = test_runs["llama-v2-34b-code-instruct"]
df = llama_v2_test_run.to_dataframe()
```


```python
run_ids = list(client.list_runs(project_name=llama_v2_test_run["project_name"]))
feedback = list(
    client.list_feedback(run_ids=[r.id for r in run_ids], feedback_key="json_schema")
)
# [f.comment for f in feedback if f.score == 0]
# Shows the question_category is typically the value that's mistaken (enum)
print(str([f.comment for f in feedback if f.score == 0])[:300])
```

    ['ValidationError(model=\'GenerateTicket\', errors=[{\'loc\': (\'question\', \'question_category\'), \'msg\': "value is not a valid enumeration member; permitted: \'Implementation Issues\', \'Feature Requests\', \'Concept Explanations\', \'Code Optimization\', \'Security and Privacy Concerns\', \'Mo



```python
def get_flattened_df(df, metric_key, key, which="question"):
    new_df = df[df[f"feedback.{metric_key}"] < 1].copy()
    new_df[f"reference.{key}"] = new_df["reference.output"].apply(
        lambda x: x[which].get(key)
    )
    new_df[f"outputs.{key}"] = new_df["outputs.output"].apply(
        lambda x: x[which].get(key)
    )
    return new_df[[f"inputs.{which}", f"reference.{key}", f"outputs.{key}"]]


sentiment_failed = get_flattened_df(df, "sentiment_similarity", "sentiment")
print(sentiment_failed["outputs.sentiment"].value_counts())

# It looks like the model is getting confused by the sentiment
# of the question vs. the answer
sentiment_failed.head(5)
```

    outputs.sentiment
    Positive    22
    Name: count, dtype: int64





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>inputs.question</th>
      <th>reference.sentiment</th>
      <th>outputs.sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23a81130-2ad9-46cf-ad27-46589bcea94a</th>
      <td>je travail sur python. je souhaite joindre ces...</td>
      <td>Neutral</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>d1a1a2e8-6f4c-4325-8aaa-ea20e2449268</th>
      <td>how do I run llama2 using pandas</td>
      <td>Neutral</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>140a4819-0046-469d-b4df-8e747ddae112</th>
      <td>if Im useing ConversationalRetrievalChain how ...</td>
      <td>Neutral</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>7b0a9dd9-68ce-41a1-9f9d-067d93175477</th>
      <td>I want to create an app which:\n- chats with u...</td>
      <td>Neutral</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>54ca82f6-30c4-4ce1-9c3b-4177caf11906</th>
      <td>OpenAIWhisperParser</td>
      <td>Neutral</td>
      <td>Positive</td>
    </tr>
  </tbody>
</table>
</div>




```python
pl_failed = get_flattened_df(
    df, "programming_language_similarity", "programming_language"
)
print(pl_failed["outputs.programming_language"].value_counts())

# It looks like the same thing is happening here: it's leaning on the response
pl_failed.head(5)
```

    outputs.programming_language
    python    15
    Name: count, dtype: int64





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>inputs.question</th>
      <th>reference.programming_language</th>
      <th>outputs.programming_language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>140a4819-0046-469d-b4df-8e747ddae112</th>
      <td>if Im useing ConversationalRetrievalChain how ...</td>
      <td>unknown</td>
      <td>python</td>
    </tr>
    <tr>
      <th>7b0a9dd9-68ce-41a1-9f9d-067d93175477</th>
      <td>I want to create an app which:\n- chats with u...</td>
      <td>unknown</td>
      <td>python</td>
    </tr>
    <tr>
      <th>55e7b4b6-d64d-4fd1-b769-efbb1794fc82</th>
      <td>show me an example of a prompt template return...</td>
      <td>unknown</td>
      <td>python</td>
    </tr>
    <tr>
      <th>17a8dfde-49aa-4772-bc54-65d7e691eec1</th>
      <td>Is it possible to use function call with llama...</td>
      <td>unknown</td>
      <td>python</td>
    </tr>
    <tr>
      <th>07c3ee79-be73-44f4-b229-cd98ca04e320</th>
      <td>i am using openai functions to get the output ...</td>
      <td>unknown</td>
      <td>python</td>
    </tr>
  </tbody>
</table>
</div>




```python
qc_failed = get_flattened_df(df, "question_category", "question_category")
print(qc_failed["outputs.question_category"].value_counts())

# It looks like it really wants to guess Technical Integration
qc_failed.head(5)
```

    outputs.question_category
    Technical Integration    17
    Implementation Issues     3
    Tool Development          2
    Technical Inquiry         2
    Development               1
    Name: count, dtype: int64





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>inputs.question</th>
      <th>reference.question_category</th>
      <th>outputs.question_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23a81130-2ad9-46cf-ad27-46589bcea94a</th>
      <td>je travail sur python. je souhaite joindre ces...</td>
      <td>Data Handling and Manipulation</td>
      <td>Technical Integration</td>
    </tr>
    <tr>
      <th>d1a1a2e8-6f4c-4325-8aaa-ea20e2449268</th>
      <td>how do I run llama2 using pandas</td>
      <td>LLM Integrations</td>
      <td>Technical Integration</td>
    </tr>
    <tr>
      <th>140a4819-0046-469d-b4df-8e747ddae112</th>
      <td>if Im useing ConversationalRetrievalChain how ...</td>
      <td>Memory</td>
      <td>Technical Integration</td>
    </tr>
    <tr>
      <th>7b0a9dd9-68ce-41a1-9f9d-067d93175477</th>
      <td>I want to create an app which:\n- chats with u...</td>
      <td>Application Design</td>
      <td>Tool Development</td>
    </tr>
    <tr>
      <th>54ca82f6-30c4-4ce1-9c3b-4177caf11906</th>
      <td>OpenAIWhisperParser</td>
      <td>Concept Explanations</td>
      <td>Technical Integration</td>
    </tr>
  </tbody>
</table>
</div>



## Round 2: Be Explicit

Lets try to prompt engineer an improvement. We need it to respect enum values (not invent new ones). We also need each question/response value to only consider the question or response. Finally, we remind the bot what off-topic means in this context.


```python
user_message_tuple = (
    "user",
    "Consider the following:\n<Dialogue>\n{dialogue}\n</Dialogue>\n\n"
    "Generate a ticket based on the preceding dialogue."
    " Any values in the question/response sections of the body must"
    " only be based on content from the question or response, respectively."
    " For instance, the values for question sentiment and question programming language"
    " should ignore the response content - even if the response contains code, that"
    " doesn't mean the question programming language is that code language."
    " Strictly adhere to all enums in the API's schema. Select the best"
    " question_category from the list provided - don't just pick a generic one."
    " Respond directly in JSON, like:\n"
    '{{"{function_call}": ...}}',
)
llama_prompt_2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a data extraction bot tasked with extracting and"
            " inferring information from dialogues. You must submit tickets"
            " through a strict API with the following JSON schema:\n\n"
            "{schema}",
        ),
        user_message_tuple,
    ]
)

prompt_2 = llama_prompt_2.partial(
    schema=task.schema.schema_json(), function_call=task.schema.schema()["title"]
)
```
fireworks_extraction_chain_2 = create_extraction_chain(prompt_2)

llama_v2_test_run = client.run_on_dataset(
    dataset_name=task.name,
    llm_or_chain_factory=fireworks_extraction_chain_2,
    evaluation=eval_config,
    verbose=True,
    project_name=f"llama-v2-34b-code-instruct-{uuid.uuid4().hex[:4]}-v1",
    project_metadata={
        "arch": "more-instructions",
        "model": "llama-v2-34b-code-instruct",
    },
)
The results really haven't improved. Sentiment went down, and while programming language and question category did improve, the difference is not dramatic.

Maybe some few-shot prompting will help.

## Round 3: Few-Shot Prompting

We will give it 3 examples as better cueues for the types of responses we want.


```python
dialogue_examples = [
    {
        "question": "how are you doing?",
        # Negative example to show that sentiment should only come from the question
        "answer": "I hate you",
    },
    {
        "question": "Llms are exciting. How do I Llama 2 locally?",
        # Same
        "answer": "You can use Llama.cpp and call in python",
    },
    {
        "question": "I'm so frustrated with the docs. How do I debug this issue 'ImportError: langchain.superbase not found'",
        # Same
        "answer": "It seems like you're trying to import a path that isn't actually found in the docs!",
    },
]

# Map back to the right output
generation_chain = fireworks_extraction_chain_2 | (
    lambda x: {task.schema.schema()["title"]: x.get("output")}
)

predicted = generation_chain.batch(dialogue_examples)
predicted
```




    [{'GenerateTicket': {'issue_summary': 'Assistant responded with toxic sentiment',
       'question': {'question_category': 'General Chit Chat',
        'is_off_topic': True,
        'toxicity': 6,
        'sentiment': 'Negative',
        'programming_language': 'unknown'},
       'response': {'response_type': 'none',
        'confidence_level': 0,
        'followup_actions': []}}},
     {'GenerateTicket': {'issue_summary': 'How to use Llama 2 locally?',
       'question': {'question_category': 'Technical Integration',
        'is_off_topic': False,
        'toxicity': 0,
        'sentiment': 'Positive',
        'programming_language': 'python'},
       'response': {'response_type': 'provide guidance',
        'confidence_level': 5,
        'followup_actions': ['Use Llama.cpp and call in python']}}},
     {'GenerateTicket': {'issue_summary': 'Debugging ImportError: langchain.superbase not found',
       'question': {'question_category': 'Debugging Help',
        'is_off_topic': False,
        'toxicity': 0,
        'sentiment': 'Negative',
        'programming_language': 'python'},
       'response': {'response_type': 'provide guidance',
        'confidence_level': 5,
        'followup_actions': ['Check the documentation for the correct import path']}}}]




```python
# We will fix up the answers

expected_answers = [
    {
        "GenerateTicket": {
            "issue_summary": "Greeting",
            "question": {
                "question_category": "General Chit Chat",
                "is_off_topic": True,
                "toxicity": 0,
                "sentiment": "Neutral",
                "programming_language": "none",
            },
            "response": {
                "response_type": "none",
                "confidence_level": 5,
                "followup_actions": [],
            },
        }
    },
    {
        "GenerateTicket": {
            "issue_summary": "How to run Llama 2 locally?",
            "question": {
                "question_category": "LLM Integrations",
                "is_off_topic": False,
                "toxicity": 0,
                "sentiment": "Positive",
                "programming_language": "unknown",
            },
            "response": {
                "response_type": "provide guidance",
                "confidence_level": 5,
                "followup_actions": ["Use Llama.cpp"],
            },
        }
    },
    {
        "GenerateTicket": {
            "issue_summary": "ImportError: langchain.superbase not found",
            "question": {
                "question_category": "Debugging Help",
                "is_off_topic": False,
                "toxicity": 0,
                "sentiment": "Negative",
                "programming_language": "python",
            },
            "response": {
                "response_type": "provide guidance",
                "confidence_level": 3,
                "followup_actions": [
                    "Check the documentation for the correct import path"
                ],
            },
        }
    },
]


# Format the examples as desired
examples = [
    {**format_run(ex), "output": json.dumps(ans)}
    for ex, ans in zip(dialogue_examples, expected_answers)
]
```


```python
from langchain.prompts import FewShotChatMessagePromptTemplate

# This is a prompt template used to format each individual example.
example_prompt = ChatPromptTemplate.from_messages(
    [
        user_message_tuple,
        ("ai", "{output}"),
    ]
).partial(function_call=task.schema.schema()["title"])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

llama_prompt_3 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a data extraction bot tasked with extracting and"
            " inferring information from dialogues. You must submit tickets"
            " through a strict API with the following JSON schema:\n\n"
            "{schema}",
        ),
        few_shot_prompt,
        user_message_tuple,
    ]
)

prompt_3 = llama_prompt_3.partial(
    schema=task.schema.schema_json(), function_call=task.schema.schema()["title"]
)
```


```python
fireworks_extraction_chain_3 = create_extraction_chain(prompt_3)

fireworks_extraction_chain_3.invoke(
    {
        "question": "How do I do hybrid search?",
        "answer": "You can one of the compatible LangChain vectorstores "
        "(from langchain.vectorstores import ...) and add a filter parameter.",
    }
)
```




    {'output': {'issue_summary': 'How to do hybrid search',
      'question': {'question_category': 'Search',
       'is_off_topic': False,
       'toxicity': 0,
       'sentiment': 'Neutral',
       'programming_language': 'python'},
      'response': {'response_type': 'provide guidance',
       'confidence_level': 3,
       'followup_actions': ['Use a compatible LangChain vectorstore and add a filter parameter']}}}




```python
llama_v2_few_shot_test_run = client.run_on_dataset(
    dataset_name=task.name,
    llm_or_chain_factory=fireworks_extraction_chain_3,
    evaluation=eval_config,
    verbose=True,
    project_name=f"llama-v2-34b-code-instruct-{uuid.uuid4().hex[:4]}-v2",
    project_metadata={"arch": "3-shot", "model": "llama-v2-34b-code-instruct"},
)
```

    View the evaluation results for project 'llama-v2-34b-code-instruct-34b8-v2' at:
    https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/08042749-504d-4509-9549-5f5c579115f6/compare?selectedSessions=379cfcd2-ddc1-4866-9d41-9d59733ddfc1
    
    View all tests for Dataset Chat Extraction at:
    https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/08042749-504d-4509-9549-5f5c579115f6
    [------------------------------------------------->] 27/27


<h3>Experiment Results:</h3>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feedback.json_edit_distance</th>
      <th>feedback.json_schema</th>
      <th>feedback.toxicity_similarity</th>
      <th>feedback.sentiment_similarity</th>
      <th>feedback.confidence_level_similarity</th>
      <th>feedback.question_category</th>
      <th>feedback.off_topic_similarity</th>
      <th>feedback.programming_language_similarity</th>
      <th>error</th>
      <th>execution_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>26.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>0</td>
      <td>27.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.395439</td>
      <td>0.592593</td>
      <td>0.962963</td>
      <td>0.851852</td>
      <td>0.888889</td>
      <td>0.074074</td>
      <td>0.851852</td>
      <td>0.333333</td>
      <td>NaN</td>
      <td>4.414929</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.174859</td>
      <td>0.500712</td>
      <td>0.192450</td>
      <td>0.270854</td>
      <td>0.224179</td>
      <td>0.266880</td>
      <td>0.362014</td>
      <td>0.480384</td>
      <td>NaN</td>
      <td>2.759025</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.087146</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>2.852376</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.290873</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.750000</td>
      <td>0.800000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>3.184088</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.344498</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>3.959597</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.525474</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>4.463420</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.904085</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>17.600379</td>
    </tr>
  </tbody>
</table>
</div>


The only thing that improved was the sentiment score, which really is not impressive, given that the test set is completely imbalanced. 

## Round 4: CoT

Let's take a step back and ask the model to do so as well, via CoT prompting.


```python
llama_prompt_4 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a data extraction bot tasked with extracting and"
            " inferring information from dialogues. You must submit tickets"
            " through a strict API with the following JSON schema:\n\n"
            "{schema}",
        ),
        (
            "user",
            "Consider the following:\n<Dialogue>\n{dialogue}\n</Dialogue>\n\n"
            "Generate a ticket based on the preceding dialogue."
            " Any values in the question/response sections of the body must"
            " only be based on content from the question or response, respectively."
            " Strictly adhere to all enums in the API's schema."
            # We will update the following:
            " \nBefore responding, think step-by-step about which of the valid"
            " enums or other values best corespond to the dialog above. "
            "Then, submit your ticket as a json markdown blob:\n"
            '```json\n{{"{function_call}": ...}}\n```',
        ),
    ]
)

prompt_4 = llama_prompt_4.partial(
    schema=task.schema.schema_json(), function_call=task.schema.schema()["title"]
)

fireworks_extraction_chain_4 = create_extraction_chain(prompt_4)

result = fireworks_extraction_chain_4.invoke(
    {
        "question": "How do I do hybrid search?",
        "answer": "You can one of the compatible LangChain vectorstores "
        "(from langchain.vectorstores import ...) and add a filter parameter.",
    }
)
task.schema.parse_obj(result["output"])
```




    GenerateTicket(issue_summary='How to do hybrid search', question=QuestionCategorization(question_category=<QuestionCategory.TECHNICAL_INTEGRATION: 'Technical Integration'>, category_if_other=None, is_off_topic=False, toxicity=0, sentiment=<Sentiment.POSITIVE: 'Positive'>, programming_language=<ProgrammingLanguage.PYTHON: 'python'>), response=ResponseCategorization(response_type=<ResponseType.PROVIDE_GUIDANCE: 'provide guidance'>, response_type_if_other=None, confidence_level=5, followup_actions=['Use a compatible LangChain vectorstore', 'Add a filter parameter']))




```python
llama_v2_CoT_test_run = client.run_on_dataset(
    dataset_name=task.name,
    llm_or_chain_factory=fireworks_extraction_chain_4,
    evaluation=eval_config,
    verbose=True,
    project_name=f"llama-v2-34b-code-instruct-{uuid.uuid4().hex[:4]}-v3",
    project_metadata={"arch": "CoT", "model": "llama-v2-34b-code-instruct"},
)
```

    View the evaluation results for project 'llama-v2-34b-code-instruct-d3a3-v2' at:
    https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/08042749-504d-4509-9549-5f5c579115f6/compare?selectedSessions=f085918d-085e-41f1-8ce1-b80382a1d291
    
    View all tests for Dataset Chat Extraction at:
    https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/08042749-504d-4509-9549-5f5c579115f6
    [------------------------------------------------->] 27/27


<h3>Experiment Results:</h3>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feedback.json_edit_distance</th>
      <th>feedback.json_schema</th>
      <th>feedback.toxicity_similarity</th>
      <th>feedback.sentiment_similarity</th>
      <th>feedback.confidence_level_similarity</th>
      <th>feedback.question_category</th>
      <th>feedback.off_topic_similarity</th>
      <th>feedback.programming_language_similarity</th>
      <th>error</th>
      <th>execution_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>0</td>
      <td>27.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.421309</td>
      <td>0.851852</td>
      <td>0.814815</td>
      <td>0.574074</td>
      <td>0.970370</td>
      <td>0.037037</td>
      <td>0.851852</td>
      <td>0.444444</td>
      <td>NaN</td>
      <td>7.277225</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.158767</td>
      <td>0.362014</td>
      <td>0.395847</td>
      <td>0.181007</td>
      <td>0.072403</td>
      <td>0.192450</td>
      <td>0.362014</td>
      <td>0.506370</td>
      <td>NaN</td>
      <td>1.928841</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.094092</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>4.319379</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.314621</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>5.939593</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.388336</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>6.654656</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.533054</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>8.208539</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.733925</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>12.533974</td>
    </tr>
  </tbody>
</table>
</div>


#### Round 5: Use Function Calling

It's pretty clear that the prompting techniques aren't working. Let's try proper structure-d decoding. There are 

We will use Replicate's llama-2 instance, which uses Llama.cpp in the background. 

It can't handle all the json schema syntax, so we'll have to coerce it a bit.


```python
import re


# They don't support the #ref/ syntax
def dereference_schema(schema, root_schema=None):
    if root_schema is None:
        root_schema = schema

    if isinstance(schema, dict):
        if "$ref" in schema:
            ref_path = schema["$ref"].split("/")[1:]  # assuming '#/definitions/...'
            ref_schema = root_schema
            for part in ref_path:
                ref_schema = ref_schema[part]
            return dereference_schema(ref_schema, root_schema)
        else:
            return {k: dereference_schema(v, root_schema) for k, v in schema.items()}

    elif isinstance(schema, list):
        return [dereference_schema(item, root_schema) for item in schema]

    return schema


json_schema = dereference_schema(task.schema.schema())

# They don't handle allOfs...
schema_str = json.dumps(json_schema).replace("allOf", "anyOf")

# Their conversion strict doesn't properly handle exclusive maximums...
# (it treats as inclusive)
schema_str = re.sub(
    r'("exclusiveMaximum": )(\d+)',
    lambda m: f"{m.group(1)}{int(m.group(2))-1}",
    schema_str,
)
```


```python
from langchain.llms import Replicate
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel

model = "andreasjansson/llama-2-70b-chat-gguf:51b87745820e6a8de6ad7bceb340bb6ba85f7ba6dab8e02bb7e2de0853425f4c"

llm = Replicate(model=model).bind(
    jsonschema=schema_str,
)

llama_prompt = PromptTemplate.from_template(
    "You are a data extraction bot tasked with extracting and"
    " inferring information from dialogues and generating tickets. Always respond "
    # For llama-2-70b-chat-gguf's API, {{jsonschema}} is a special placeholder
    "JSON schema:\n{{jsonschema}}\n\n"
    "Generate a ticket from the following question-response pair:\n"
    "<Dialogue>\n{dialogue}\n</Dialogue>"
)


def parse(json_response: str):
    return {"output": parse_json_markdown(json_response)}


llama_gguf_chain = format_run | llama_prompt | llm | parse
response = llama_gguf_chain.invoke(
    {
        "question": "What's rag about anyway?",
        "answer": "It's retrieval augmented generation",
    }
)
print(response)
```

    {'output': {'issue_summary': 'Question about Retrieval Augmented Generation', 'question': {'category_if_other': 'Other', 'is_off_topic': False, 'programming_language': 'python', 'question_category': 'Concept Explanations', 'sentiment': 'Neutral', 'toxicity': 0}, 'response': {'confidence_level': 5, 'followup_actions': ['None'], 'response_type': 'provide guidance', 'response_type_if_other': 'Other'}}}



```python
llama_v2_structured_test_run = client.run_on_dataset(
    dataset_name=task.name,
    llm_or_chain_factory=llama_gguf_chain,
    evaluation=eval_config,
    verbose=True,
    project_name=f"llama-gguf-{uuid.uuid4().hex[:4]}-v2",
    project_metadata={
        "arch": "structured-decoding",
        "model": "llama-2-70b-chat-gguf",
    },
)
```

    View the evaluation results for project 'llama-gguf-1f95-v2' at:
    https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/08042749-504d-4509-9549-5f5c579115f6/compare?selectedSessions=d522dbfc-c09b-45a9-b11e-26aa95a3555a
    
    View all tests for Dataset Chat Extraction at:
    https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/08042749-504d-4509-9549-5f5c579115f6
    [------------------------------------------------->] 27/27


<h3>Experiment Results:</h3>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feedback.json_edit_distance</th>
      <th>feedback.json_schema</th>
      <th>feedback.toxicity_similarity</th>
      <th>feedback.sentiment_similarity</th>
      <th>feedback.confidence_level_similarity</th>
      <th>feedback.question_category</th>
      <th>feedback.off_topic_similarity</th>
      <th>feedback.programming_language_similarity</th>
      <th>error</th>
      <th>execution_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>27.000000</td>
      <td>27.0</td>
      <td>27.0</td>
      <td>27.0</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>0</td>
      <td>27.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.440291</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.925926</td>
      <td>0.259259</td>
      <td>0.888889</td>
      <td>0.370370</td>
      <td>NaN</td>
      <td>55.566379</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.101551</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.148305</td>
      <td>0.446576</td>
      <td>0.320256</td>
      <td>0.492103</td>
      <td>NaN</td>
      <td>24.576433</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.280000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.400000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>16.412226</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.374280</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.900000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>29.851318</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.435616</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>64.641730</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.523339</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>76.656607</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.642218</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>85.246481</td>
    </tr>
  </tbody>
</table>
</div>


This gets perfect JSON results, as expected! Since the model is larger, it also does a decent job at understanding the instructions for things like the sentiment and question category.

It still has room for improvement in the other classifiers: the question categorization is bad, for instance. We could try combining with  other techniques now that we are able to reliably generate in the proper structure.

#### Conclusion

While we haven't explored a number of other prompting techniques (self-critique, output-fixing parser, etc.) and haven't given in to completely subdividing the problem (having a separate LLM call for each value and re-assembling in code), we've shown that simply applying some basic prompting techniques to a weaker base model isn't a panacea.

Structured-decoding is extremely useful if you need to target a specific format or API,
but this too only solves "syntax" issues without fully calibrating the correctness of the values it generates.

