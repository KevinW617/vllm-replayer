# Force expected response

Give vLLM an expected response, vllm-replayer will fully replay the response with 0 overhead. Useful for Agent / LLM traj replay to align the performance of serving engine.

Make vLLM emit a fixed token sequence instead of sampling. Used for replay or prefix-cache alignment: the next request (e.g. same session) sees the exact assistant text you specified, so KV cache and continuations stay consistent.

## API

Send a chat completion with `vllm_xargs`:

| Key             | Type  | Description |
|-----------------|-------|-------------|
| `expected_resp` | string| Exact assistant text to output (will be tokenized server-side). |
| `force_expected`| bool  | If `true`, override sampled tokens with `expected_resp` step-by-step. Default `false`. |

- If `force_expected` is `false`, `expected_resp` is ignored.
- If `force_expected` is `true` and `expected_resp` is omitted, no override (same as `false`).

## Example

```json
POST /v1/chat/completions
{
  "model": "...",
  "messages": [{"role": "user", "content": "..."}],
  "vllm_xargs": {
    "expected_resp": "Exact text to return.",
    "force_expected": true
  }
}
```

See `examples/force_expected_completion.py` for a runnable script.

## Server behavior (V1 engine)

1. **Processor**: When `force_expected` is true and `expected_resp` is set, the server tokenizes `expected_resp` (same tokenizer as the request, including LoRA) and stores token ids in sampling `extra_args`.
2. **Model runner**: After sampling each step, if the request has `force_expected` and `expected_token_ids`, the runner overwrites the sampled token(s) for that step with the corresponding expected token(s). KV cache and subsequent inputs therefore match the expected sequence.

Logprobs and stop checks still reflect the original sampled token; only the emitted token sequence is forced. Speculative decoding is not coordinated with this path.
