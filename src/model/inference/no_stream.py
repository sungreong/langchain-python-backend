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
        print("---prompt---")
        print(prompt)
        print("------------")
        output_text = llm.invoke(prompt)
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


from typing import List, Optional
from ...utils.openai_types import ChatMessage, Function, Tool
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.completion_create_params import CompletionCreateParams, CompletionCreateParamsStreaming
from ...utils.openai_types import (
    ChatCompletionResponseChoice,
    ChatCompletionResponse,
)


def generate_chat_message(
    messages: List[ChatMessage],
    model,
    completion_id,
    prompt_list: list,
    llm: HuggingFacePipeline,
    functions: Optional[List[Function]] = None,
    tools: Optional[List[Tool]] = None,
    **kwargs,
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
        output_text = llm.invoke(prompt)
        list_prompt_tokens = llm.pipeline.tokenizer.encode(prompt)
        list_completion_tokens = llm.pipeline.tokenizer.encode(output_text)
        prompt_tokens += len(list_prompt_tokens)
        completion_tokens += len(list_completion_tokens)
        total_tokens += len(list_prompt_tokens + list_completion_tokens)
        choices = [
            ChatCompletionResponseChoice(
                finish_reason="stop",
                index=idx,
                message=ChatMessage(
                    role="ai",  # messages[-1]["role"],
                    # TODO: Implement message_id(로직 찾기)
                    content=output_text,
                    function_call=None,
                    tool_calls=None,
                    tool_call_id=None,
                    name=None,
                ),
            )
        ]
    else:
        return ChatCompletionResponse(
            model=model,
            id=completion_id,
            object="chat.completion",
            created=int(time.time()),
            choices=choices,
            usage=CompletionUsage(
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
            ),
        )
