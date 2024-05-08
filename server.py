import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_CACHE"] = "C:/Users/leesu/Project/ReactJS/ai-assistant/llm-python-backend/model_list"


import argparse
import json
import uuid
from typing import Union

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

## custom
from src.utils.openai_types import (
    ChatInput,
)
from src.utils.openai_completion_types import (
    CompletionCreateParams,
)
from src.model.load.pipeline import define_pipeline
from src.model.load.pipeline import load_model, unload_model
from src.model.inference.no_stream import generate_message
from src.model.inference.stream import generate_stream
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
    request_id = str(uuid.uuid4())
    llm, streamer = define_pipeline(request)
    completion = generate_message(request["model"], request_id, request["prompt"], llm)
    completion_dic = completion.dict(exclude_unset=True)
    return json.dumps(completion_dic, ensure_ascii=False)


@app.delete("/model/unload/")
def unload_model_endpoint(model_id: UnLoadModelRequest):
    result = unload_model(model_id)
    return {"message": result}


@app.post("/v1/chat/completions")
async def chat_endpoint(chat_input: ChatInput):
    raise NotImplementedError("Not implemented yet!")


@app.post("/v1/completions")
async def generate_text(completion_input: CompletionCreateParams):
    llm, streamer = define_pipeline(completion_input)
    request_id = str(uuid.uuid4())

    if completion_input["stream"]:

        response_generator = generate_stream(
            model=completion_input["model"],
            request_id=request_id,
            prompt_list=completion_input["prompt"],
            llm=llm,
            streamer=streamer,
        )

        def get_response_stream():
            for response in response_generator:
                chunk_dic = response.dict(exclude_unset=True)
                chunk_data = json.dumps(chunk_dic, ensure_ascii=False)
                yield f"data: {chunk_data}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(get_response_stream(), media_type="text/event-stream")

    else:
        completion = generate_message(completion_input["model"], request_id, completion_input["prompt"], llm)
        return completion


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
