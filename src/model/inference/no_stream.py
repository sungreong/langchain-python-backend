from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from ...utils.openai_completion_types import (
    CompletionChoice,
    CompletionUsage,
    Completion,
)
import time


def generate_message(model, completion_id, prompt_list: list[str], llm: HuggingFacePipeline):
    choices = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for idx, prompt in enumerate(prompt_list):
        print(prompt)
        output_text = llm.invoke(prompt)
        print(output_text)
        list_prompt_tokens = llm.pipeline.tokenizer.encode(prompt)
        list_completion_tokens = llm.pipeline.tokenizer.encode(output_text)
        prompt_tokens += len(list_prompt_tokens)
        completion_tokens += len(list_completion_tokens)
        total_tokens += len(list_prompt_tokens + list_completion_tokens)
        choices.append(
            CompletionChoice(
                finish_reason="stop",
                index=idx,
                text=output_text,
                logprobs=None,
            )
        )
    else:
        return Completion(
            model=model,
            id=completion_id,
            object="completion",
            created=int(time.time()),
            choices=choices,
            usage=CompletionUsage(
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
            ),
        )
