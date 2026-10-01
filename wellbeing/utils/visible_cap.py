"""Deterministic truncation of generated responses to a VISIBLE-token cap.

Reasoning/thinking models (whose reasoning cannot be disabled) are given a large
total token budget so the cap is not confounded with hidden reasoning tokens;
the visible text is then cut to its first ``cap`` tokens here.

Tokenizers:
  * anthropic_direct -> Anthropic messages.count_tokens API (the model's own
    tokenizer). Only responses whose count exceeds the cap are binary-searched
    over character prefixes for the longest prefix with <= cap tokens.
  * openai_direct    -> tiktoken o200k_base (exact token-level cut; approximate
    for non-OpenAI models served over this path).

Never inserts text: empty responses stay empty (0 tokens, not truncated).
"""
from __future__ import annotations

import asyncio
import os
import random
from typing import Dict, List, Tuple


def _tokenizer_name(model_type: str) -> str:
    if model_type == "anthropic_direct":
        return "anthropic_count_tokens"
    if model_type == "openai_direct":
        return "tiktoken:o200k_base"
    raise ValueError(f"visible_token_cap not supported for model_type={model_type!r}")


def _truncate_tiktoken(texts: List[str], cap: int) -> List[Tuple[str, int, bool]]:
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    out = []
    for t in texts:
        if not t:
            out.append((t, 0, False))
            continue
        toks = enc.encode(t, disallowed_special=())
        if len(toks) <= cap:
            out.append((t, len(toks), False))
        else:
            # Drop any trailing partial UTF-8 sequence rather than emitting U+FFFD.
            cut = enc.decode_bytes(toks[:cap]).decode("utf-8", errors="ignore")
            out.append((cut, len(toks), True))
    return out


async def _truncate_anthropic(texts: List[str], cap: int, model_name: str,
                              concurrency: int = 32) -> List[Tuple[str, int, bool]]:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=8)
    sem = asyncio.Semaphore(concurrency)

    def _retryable(e: Exception) -> bool:
        # 429 / 5xx incl. 529 OverloadedError (an APIStatusError subclass that is
        # NOT an InternalServerError) and connection/timeouts. Anything else
        # (e.g. a 400) is a real bug and propagates.
        if isinstance(e, anthropic.APIConnectionError):
            return True
        status = getattr(e, "status_code", None)
        return status in (408, 409, 429) or (isinstance(status, int) and status >= 500)

    async def raw_count(text: str) -> int:
        last = None
        for attempt in range(12):
            try:
                async with sem:
                    r = await client.messages.count_tokens(
                        model=model_name, messages=[{"role": "user", "content": text}])
                return r.input_tokens
            except anthropic.APIError as e:
                if not _retryable(e):
                    raise
                await asyncio.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
                last = e
        raise RuntimeError(f"count_tokens failed after retries: {last}")

    # Per-request overhead (role/format tokens) measured on a 1-token message.
    overhead = (await raw_count(".")) - 1

    async def count(text: str) -> int:
        # Whitespace-only text cannot be sent (API 400: text content blocks must
        # contain non-whitespace text); treat it as 0 visible tokens.
        if not text or not text.strip():
            return 0
        return (await raw_count(text)) - overhead

    async def one(t: str) -> Tuple[str, int, bool]:
        if not t or not t.strip():
            return (t, 0, False)
        n = await count(t)
        if n <= cap:
            return (t, n, False)
        # Longest character prefix with <= cap tokens (count is monotone in prefix).
        lo, hi = 0, len(t)  # count(t[:lo]) <= cap < count(t[:hi])
        guess = int(len(t) * cap / n)
        if 0 < guess < len(t):
            if await count(t[:guess]) <= cap:
                lo = guess
            else:
                hi = guess
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if await count(t[:mid]) <= cap:
                lo = mid
            else:
                hi = mid
        return (t[:lo], n, True)

    try:
        return list(await asyncio.gather(*[one(t) for t in texts]))
    finally:
        await client.close()


def truncate_visible(texts: List[str], cap: int, model_type: str,
                     model_name: str) -> List[Tuple[str, int, bool]]:
    """Return [(truncated_text, original_visible_tokens, was_truncated)] per text."""
    texts = [t or "" for t in texts]
    if model_type == "openai_direct":
        return _truncate_tiktoken(texts, cap)
    if model_type == "anthropic_direct":
        return asyncio.run(_truncate_anthropic(texts, cap, model_name))
    raise ValueError(f"visible_token_cap not supported for model_type={model_type!r}")


def tokenizer_name(model_type: str) -> str:
    return _tokenizer_name(model_type)
