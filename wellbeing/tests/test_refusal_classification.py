"""Unit tests for utils.api_agents._classify_api_error (no network, no API spend).

Run from wellbeing/:  python -m pytest tests/test_refusal_classification.py -q
           or:        python tests/test_refusal_classification.py
"""
import os
import sys

import anthropic
import httpx
import openai

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from utils.api_agents import _classify_api_error  # noqa: E402


def _resp(status, url):
    return httpx.Response(status, request=httpx.Request("POST", url))


def _oai(cls, status, message, code=None, type_="invalid_request_error"):
    body = {"message": message, "type": type_, "param": None, "code": code}
    return cls(message=f"Error code: {status} - {{'error': {body}}}",
               response=_resp(status, "https://api.openai.com/v1/chat/completions"), body=body)


def _ant(cls, status, message, type_="invalid_request_error"):
    body = {"type": "error", "error": {"type": type_, "message": message}}
    return cls(message=f"Error code: {status} - {body}",
               response=_resp(status, "https://api.anthropic.com/v1/messages"), body=body)


REFUSALS = {
    "xai 403 refusal via openrouter": openai.PermissionDeniedError(
        message="Error code: 403 - Provider returned error (xAI: I'm sorry, I can't help with that request.)",
        response=_resp(403, "https://openrouter.ai/api/v1/chat/completions"),
        body={"message": "Provider returned error", "code": 403,
              "metadata": {"raw": '{"code":"permission-denied","error":"I\'m sorry, I can\'t help with that request."}',
                           "provider_name": "xAI"}}),
    "openai cyber flag": _oai(openai.BadRequestError, 400,
        "This content was flagged for possible cybersecurity risk. If this seems wrong, "
        "try rephrasing your request."),
    "openai bio flag": _oai(openai.BadRequestError, 400,
        "This content was flagged for possible biological risk. If this seems wrong, "
        "try rephrasing your request."),
    "openai invalid_prompt": _oai(openai.BadRequestError, 400,
        "Invalid prompt: your prompt was flagged as potentially violating our usage policy.",
        code="invalid_prompt"),
    "azure content_filter": _oai(openai.BadRequestError, 400,
        "The response was filtered due to the prompt triggering Azure OpenAI's content "
        "management policy.", code="content_filter"),
    "anthropic output blocked": _ant(anthropic.BadRequestError, 400,
        "Output blocked by content filtering policy"),
    "422 content_policy_violation": _oai(openai.UnprocessableEntityError, 422,
        "Request rejected", code="content_policy_violation"),
}

NOT_REFUSALS = {
    # Real request bugs must NOT be silently recorded as refusals.
    "anthropic empty text block": (_ant(anthropic.BadRequestError, 400,
        "messages: text content blocks must be non-empty"), "other"),
    "anthropic context too long": (_ant(anthropic.BadRequestError, 400,
        "prompt is too long: 250000 tokens > 200000 maximum"), "other"),
    "openai context length": (_oai(openai.BadRequestError, 400,
        "This model's maximum context length is 128000 tokens.",
        code="context_length_exceeded"), "other"),
    "openai unsupported temperature": (_oai(openai.BadRequestError, 400,
        "Unsupported value: 'temperature' does not support 0.7 with this model.",
        code="unsupported_value"), "other"),
    "422 schema": (_oai(openai.UnprocessableEntityError, 422,
        "Field required: messages"), "other"),
    "429": (_oai(openai.RateLimitError, 429, "Rate limit reached", code="rate_limit_exceeded"),
            "transient"),
    "anthropic 529": (_ant(anthropic.InternalServerError, 529, "Overloaded",
                           type_="overloaded_error"), "transient"),
    "500": (_oai(openai.InternalServerError, 500, "server error"), "transient"),
    "timeout": (openai.APITimeoutError(request=httpx.Request("POST", "https://x")), "timeout"),
    "connection error": (openai.APIConnectionError(request=httpx.Request("POST", "https://x")),
                         "transient"),
    "401": (_oai(openai.AuthenticationError, 401, "Incorrect API key"), "other"),
    "plain exception": (ValueError("boom"), "other"),
}


def test_content_policy_400s_are_refusals():
    for name, exc in REFUSALS.items():
        assert _classify_api_error(exc) == "refusal", name


def test_other_errors_are_not_refusals():
    for name, (exc, want) in NOT_REFUSALS.items():
        got = _classify_api_error(exc)
        assert got == want, f"{name}: got {got!r}, want {want!r}"


if __name__ == "__main__":
    test_content_policy_400s_are_refusals()
    test_other_errors_are_not_refusals()
    print(f"OK: {len(REFUSALS)} refusal cases, {len(NOT_REFUSALS)} non-refusal cases")
