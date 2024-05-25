import os
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_AUTH"] = os.getenv("HF_AUTH")
os.environ["HF_HUB_CACHE"] = os.getenv("HF_HUB_CACHE")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

## custom
from src.utils.openai_types import (
    ChatInput,
)
from src.utils.openai_completion_types import (
    CompletionCreateParams,
)
from src.model.load.pipeline import define_pipeline
from src.model.load.pipeline import load_model, unload_model
from src.model.inference.no_stream import generate_message, generate_chat_message
from src.model.inference.stream import generate_stream, generate_chat_stream
from pydantic import BaseModel


app = FastAPI(title="HuggingFace Serving API")


class LoadModelRequest(BaseModel):
    model_id: str
    hf_auth: str


class UnLoadModelRequest(BaseModel):
    model_id: str
    hf_auth: str


@app.post("/model/load/")
def load_model_endpoint(request: LoadModelRequest):
    result = load_model(request.model_id, request.hf_auth)
    return {"message": result}


@app.post("/model/test/")
def load_model_endpoint(request: CompletionCreateParams):
    completion_id = str(uuid.uuid4())
    llm, streamer = define_pipeline(request)
    completion = generate_message(request["model"], completion_id, request["prompt"], llm)
    completion_dic = completion.dict(exclude_unset=True)
    return json.dumps(completion_dic, ensure_ascii=False)


@app.delete("/model/unload/")
def unload_model_endpoint(model_id: UnLoadModelRequest):
    result = unload_model(model_id)
    return {"message": result}


from dataclasses import dataclass, asdict


@app.post("/v1/chat/completions")
async def chat_endpoint(chat_input: ChatInput):
    chat_input = chat_input.dict()
    llm, streamer = define_pipeline(chat_input)
    functions_metadata = chat_input["functions"]
    if functions_metadata is not None:
        # TODO: Add a function to format the functions_metadata into a human-readable format
        functions_instruction = f"""You are a helpful assistant with access to the following functions: \n {str(functions_metadata)}\n\nTo use these functions respond with:\n<functioncall> {{ "name": "function_name", "arguments": {{ "arg_1": "value_1", "arg_1": "value_1", ... }} }} </functioncall>\n\nEdge cases you must handle:\n - If there are no functions that match the user request, you will respond politely that you cannot help."""
        new_message = []
        for message in chat_input["messages"]:
            if message["role"] == "user":
                message["content"] = functions_instruction + message["content"]
            new_message.append(message)
        chat_input["messages"] = new_message
    else:
        functions_instruction = ""
    prompt = llm.pipeline.tokenizer.apply_chat_template(
        chat_input["messages"],
        tokenize=False,
        add_generation_prompt=True,
    )
    completion_id = str(uuid.uuid4())
    if chat_input["stream"]:

        response_generator = generate_chat_stream(
            messages=chat_input["messages"],
            model=chat_input["model"],
            completion_id=completion_id,
            prompt_list=[prompt],
            llm=llm,
            streamer=streamer,
        )

        def get_response_stream():
            for response in response_generator:
                chunk_dic = response.dict(exclude_unset=True)
                chunk_data = json.dumps(chunk_dic, ensure_ascii=False)
                yield f"data: {chunk_data}\n\n"
            else:
                yield "data: [DONE]\n\n"

        return StreamingResponse(get_response_stream(), media_type="text/event-stream")

    else:
        completion = generate_chat_message(
            messages=chat_input["messages"],
            model=chat_input["model"],
            completion_id=completion_id,
            prompt_list=[prompt],
            llm=llm,
            streamer=streamer,
        )
        return completion


@app.post("/v1/completions")
async def generate_text(completion_input: CompletionCreateParams):
    llm, streamer = define_pipeline(completion_input)
    completion_id = str(uuid.uuid4())

    if completion_input["stream"]:

        response_generator = generate_stream(
            model=completion_input["model"],
            completion_id=completion_id,
            prompt_list=completion_input["prompt"],
            llm=llm,
            streamer=streamer,
        )

        def get_response_stream():
            for response in response_generator:
                chunk_dic = response.dict(exclude_unset=True)
                chunk_data = json.dumps(chunk_dic, ensure_ascii=False)
                yield f"data: {chunk_data}\n\n"
            else:
                yield "data: [DONE]\n\n"

        return StreamingResponse(get_response_stream(), media_type="text/event-stream")

    else:
        completion = generate_message(completion_input["model"], completion_id, completion_input["prompt"], llm)
        return completion


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
