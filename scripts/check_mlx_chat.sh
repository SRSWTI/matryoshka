#!/usr/bin/env bash
set -euo pipefail

curl -sS http://127.0.0.1:44445/v1/chat/completions \
  -H "Authorization: Bearer 2508" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MercuriusDream--Qwen3.5-4B-MLX-mxfp8",
    "messages": [{"role": "user", "content": "what is 1+1"}],
    "max_tokens": 500,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
