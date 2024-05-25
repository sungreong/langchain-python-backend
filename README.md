

# write .env file

```
HF_AUTH=hf-xxx # 입력
HF_HUB_CACHE= # 경로 입력 (원하는 경로에 입력)
```


# fastapi 실행

```
python server.py
```


# model load 

```
curl -X 'POST' \
  'http://localhost:8000/model/load/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model_id": "microsoft/Phi-3-mini-4k-instruct",
  "hf_auth": "##"
}'
```


# 테스트(example/open_ai_api_test.ipynb)

- 해당 파일에서 모델을 불러와서 호출 가능
- `model_name` 은 load_model 과 동일해야 함

```
from langchain_openai import ChatOpenAI, OpenAI
llm = OpenAI(
    model_name="microsoft/Phi-3-mini-4k-instruct",  # Ensure this matches a model your server can serve
    openai_api_base="http://localhost:8000/v1",  # Change this to your local server's URL
    openai_api_key="Not needed for local server",  # API key not required for local setups
    openai_proxy="",  # Endpoint for generating responses
    temperature=0.3,  # Deterministic output,
    max_tokens=3000,  # Maximum tokens to generate
    stream=False,
    frequency_penalty=0.0,
    top_p=1.0,
)

```

- invoke

```
result = llm.invoke("write python code to add two numbers")
print(result)
```

- stream

```
for i in llm.stream("write python code to add two numbers"):
    print(i,end="",flush=True)
```