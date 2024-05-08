from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from ...utils.openai_completion_types import (
    CompletionChoice,
    CompletionUsage,
    Completion,
)
import time

from typing import List, Optional, Union, Dict
from transformers import pipeline, TextIteratorStreamer


from threading import Thread
from langchain_core.language_models import llms


def generate(llm: HuggingFacePipeline, query: str):
    llm.invoke(input=query)


def start_generation(llm: HuggingFacePipeline, query: str):
    # Creating a thread with generate function as a target
    thread = Thread(target=generate, kwargs={"query": query, "llm": llm})
    # Starting the thread
    thread.start()
    return thread


def generate_stream(model, request_id, prompt_list: list, llm: HuggingFacePipeline, streamer: TextIteratorStreamer):
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    if isinstance(prompt_list, str):
        prompt_list = [prompt_list]
    for idx, prompt in enumerate(prompt_list):
        print(prompt)
        list_prompt_tokens = llm.pipeline.tokenizer.encode(prompt)
        prompt_tokens += len(list_prompt_tokens)
        thread = start_generation(llm, prompt)
        for output_text in streamer:
            list_completion_tokens = llm.pipeline.tokenizer.encode(output_text)
            completion_tokens += len(list_completion_tokens)
            total_tokens += len(list_prompt_tokens + list_completion_tokens)
            complete_usage = CompletionUsage(
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
            )
            choices = [
                CompletionChoice(
                    finish_reason=None,
                    index=idx,
                    text=output_text,
                    logprobs=None,
                )
            ]
            chunk = Completion(
                model=model,
                id=request_id,
                object="completion",
                created=int(time.time()),
                choices=choices,
                usage=complete_usage,
            )
            yield chunk
        else:
            choices = [
                CompletionChoice(
                    finish_reason="stop",
                    index=idx,
                    text="",
                    logprobs=None,
                )
            ]
            chunk = Completion(
                model=model,
                id=request_id,
                object="completion",
                created=int(time.time()),
                choices=choices,
                usage=complete_usage,
            )
            yield chunk
            thread.join()
