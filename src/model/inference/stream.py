from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from ...utils.openai_completion_types import (
    CompletionChoice,
    CompletionUsage,
    Completion,
)
import time
from transformers import TextIteratorStreamer

from threading import Thread
from langchain_core.language_models import llms
from typing import List, Optional
from ...utils.openai_types import ChatMessage, Function, Tool


def generate(llm: HuggingFacePipeline, query: str):
    llm.invoke(input=query)


def start_generation(llm: HuggingFacePipeline, query: str):
    # Creating a thread with generate function as a target
    thread = Thread(target=generate, kwargs={"query": query, "llm": llm})
    # Starting the thread
    thread.start()
    return thread


def generate_stream(
    model,
    completion_id,
    prompt_list: list,
    llm: HuggingFacePipeline,
    streamer: TextIteratorStreamer,
    functions: Optional[List[Function]] = None,
    tools: Optional[List[Tool]] = None,
    stop=[],
):
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    if isinstance(prompt_list, str):
        prompt_list = [prompt_list]
    for idx, prompt in enumerate(prompt_list):
        print("---prompt---")
        print(prompt)
        print("------------")
        list_prompt_tokens = llm.pipeline.tokenizer.encode(prompt)
        prompt_tokens += len(list_prompt_tokens)
        thread = start_generation(llm, prompt)
        total_text = ""
        for output_text in streamer:
            total_text += output_text
            print(total_text, stop)
            if any([s in total_text for s in stop]):
                break
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
                id=completion_id,
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
                id=completion_id,
                object="completion",
                created=int(time.time()),
                choices=choices,
                usage=complete_usage,
            )
            yield chunk
            thread.join()


from ...utils.openai_types import (
    ChatCompletionResponseStreamChoice,
    DeltaMessage,
    ChatCompletionStreamResponse,
)


def generate_chat_stream(
    messages,
    model,
    completion_id,
    prompt_list: list,
    llm: HuggingFacePipeline,
    streamer: TextIteratorStreamer,
    functions: Optional[List[Function]] = None,
    tools: Optional[List[Tool]] = None,
    stop=[],
):
    # TODO: Implement functions and tools
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    if isinstance(prompt_list, str):
        prompt_list = [prompt_list]
    for idx, prompt in enumerate(prompt_list):
        print("---prompt---")
        print(prompt)
        print("------------")
        list_prompt_tokens = llm.pipeline.tokenizer.encode(prompt)
        prompt_tokens += len(list_prompt_tokens)
        thread = start_generation(llm, prompt)
        total_text = ""
        for output_text in streamer:
            total_text += output_text
            print(output_text, end="", flush=True)
            if any([s in total_text for s in stop]):
                break
            list_completion_tokens = llm.pipeline.tokenizer.encode(output_text)
            completion_tokens += len(list_completion_tokens)
            total_tokens += len(list_prompt_tokens + list_completion_tokens)
            complete_usage = CompletionUsage(
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
            )
            choices = [
                ChatCompletionResponseStreamChoice(
                    finish_reason=None,
                    index=idx,
                    delta=DeltaMessage(
                        role="ai",  # messages[-1]["role"],
                        content=output_text,
                    ),
                )
            ]
            chunk = ChatCompletionStreamResponse(
                model=model,
                id=completion_id,
                # object="chat.completion.chunk",
                created=int(time.time()),
                choices=choices,
                usage=complete_usage,
            )
            yield chunk
        else:
            choices = [
                ChatCompletionResponseStreamChoice(
                    finish_reason="stop",
                    index=idx,
                    delta=DeltaMessage(
                        role="ai",  # messages[-1]["role"],
                        content="",
                    ),
                )
            ]
            chunk = ChatCompletionStreamResponse(
                model=model,
                id=completion_id,
                # object="chat.completion",
                created=int(time.time()),
                choices=choices,
                usage=complete_usage,
            )
            yield chunk
            thread.join()
