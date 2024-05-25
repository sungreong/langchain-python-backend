@echo off
set URL=http://localhost:8000/model/load/
set MODEL_ID=microsoft/Phi-3-mini-4k-instruct
set HF_AUTH=
set HF_HUB_CACHE=C:/Users/leesu/Project/ReactJS/ai-assistant/llm-python-backend/model_list
curl -X POST "%URL%" -H "Content-Type: application/json" -d "{\"model_id\": \"%MODEL_ID%\", \"hf_auth\": \"%HF_AUTH%\"}"

echo Success, press any key to exit...
pause 
