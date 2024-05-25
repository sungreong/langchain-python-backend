from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline, TextIteratorStreamer
from ...utils.openai_completion_types import CompletionCreateParams
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline


models = {}  # Dictionary to store model instances


def unload_model(model_id):
    if model_id in models:
        del models[model_id]
        return "Model unloaded"
    else:
        return "Model not found"


def load_model(model_id, hf_auth):
    print("Loading model", model_id)
    model_config = AutoConfig.from_pretrained(
        model_id,
        token=hf_auth,
        trust_remote_code=True,
        # cache_dir=cache_dir,
    )
    print("Model config loaded")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=model_config,
        quantization_config=None,  # Define `bnb_config` if needed
        device_map="auto",
        token=hf_auth,
        trust_remote_code=True,
        # cache_dir=cache_dir,
    )
    print("Model loaded")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_auth,
        trust_remote_code=True,
        # cache_dir=cache_dir,
    )
    print("Tokenizer loaded")
    models[model_id] = dict(model=model, tokenizer=tokenizer)
    return "Model loaded"


def get_model(model_id):
    print("get model")
    return models.get(model_id)


def define_pipeline(request: CompletionCreateParams):
    model_info = get_model(request["model"])
    print("Load model")
    # if not llm:
    #     raise HTTPException(status_code=404, detail="Model not loaded")
    if request["stream"]:
        print("use streamer...")
        streamer = TextIteratorStreamer(model_info["tokenizer"], skip_prompt=True, timeout=None)
    else:
        streamer = None
    print("pipeline", request)
    generate_text = pipeline(
        "text-generation",
        model=model_info["model"],
        tokenizer=model_info["tokenizer"],
        do_sample=False,
        return_full_text=False,
        temperature=request.get("temperature", 0.3),
        max_new_tokens=request.get("max_tokens", 10),
        # repetition_penalty=request.get("presence_penalty", 1.1),
        use_fast=True,
        top_p=request.get("top_p", 0.95),
        streamer=streamer if request["stream"] else None,
    )
    print("Pipeline loaded")
    llm = HuggingFacePipeline(pipeline=generate_text)
    print(request)
    if request.get("stop", None) is not None:
        llm = llm.bind(stop=request.get("stop"))
    return llm, streamer
