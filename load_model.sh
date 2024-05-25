#!/bin/bash

curl -X 'POST' \
    'http://localhost:8000/model/load/' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "model_id": "microsoft/Phi-3-mini-4k-instruct",
    "hf_auth": ""
}'