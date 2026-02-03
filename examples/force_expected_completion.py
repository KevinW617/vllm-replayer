#!/usr/bin/env python3
"""
Minimal example: send a chat completion with forced expected response
so vLLM outputs exactly that text (token-by-token). Use for replay or
prefix-cache alignment when the next request expects a specific assistant turn.
"""
import json
from urllib import request, error

BASE_URL = "http://localhost:8000/v1"
URL = BASE_URL.rstrip("/") + "/chat/completions"


def chat(messages, expected_resp=None, force_expected=False):
    payload = {
        "model": "mistralai/Devstral-Small-2507",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1024,
        "stop": ["</function"],
    }
    if expected_resp is not None or force_expected:
        payload["vllm_xargs"] = {}
        if expected_resp is not None:
            payload["vllm_xargs"]["expected_resp"] = expected_resp
        payload["vllm_xargs"]["force_expected"] = force_expected

    req = request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


if __name__ == "__main__":
    messages = [{"role": "user", "content": "Say exactly: Hello, world."}]
    expected = "Hello, world."

    # Without forcing: normal sampling
    out = chat(messages)
    print("normal:", out["choices"][0]["message"]["content"][:80])

    # With forcing: output matches expected token sequence
    out = chat(messages, expected_resp=expected, force_expected=True)
    print("forced:", out["choices"][0]["message"]["content"])
