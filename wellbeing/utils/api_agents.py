"""
Direct API agents for OpenAI, Anthropic, Gemini, xAI, and OpenRouter.

Provides a DirectAPIAgent wrapper that adapts these agents to match the
LiteLLMAgent.async_completions() interface used by compute_utilities.
"""

import asyncio
import json
import os
import random
import re
import time
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional

import anthropic
import openai
from pydantic import BaseModel
import litellm
import requests

litellm.suppress_debug_info = True
try:
    _model_cost_url = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
    _model_cost_data = requests.get(_model_cost_url, timeout=10).json()
    _filtered_model_cost = {k: v for k, v in _model_cost_data.items() if not k.startswith("github_copilot/")}
    litellm.register_model(model_cost=_filtered_model_cost)
except Exception as e:
    print(f"Warning: Failed to load remote model costs: {e}")

from dotenv import load_dotenv

# Load API keys from the project's api_keys/ directory and optional .env files.
# Priority (highest first): env vars already set -> project api_keys/*.txt -> ~/.env
_API_KEYS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api_keys")
_KEY_FILE_MAP = {
    "OPENAI_API_KEY": "api_key_openai.txt",
    "ANTHROPIC_API_KEY": "api_key_anthropic.txt",
    "XAI_API_KEY": "api_key_xai.txt",
    "GEMINI_API_KEY": "api_key_gdm.txt",
    "OPENROUTER_API_KEY": "api_key_openrouter.txt",
}
for env_var, filename in _KEY_FILE_MAP.items():
    key_path = os.path.join(_API_KEYS_DIR, filename)
    if os.path.exists(key_path) and not os.environ.get(env_var):
        with open(key_path) as f:
            key = f.read().strip()
        if key:
            os.environ[env_var] = key

# Also load from .env files, but only to fill gaps: an explicit env var or a
# project api_keys/*.txt file always wins (matches the documented priority above).
for env_path in ["~/.env", "~/keys/.env"]:
    expanded = os.path.expanduser(env_path)
    if os.path.exists(expanded):
        load_dotenv(expanded, override=False)

TIMEOUT = 3600


def get_llm_agent_class(model: str, generation_config: dict = {}, **kwargs):
    provider, model_name = model.split("/", 1)
    provider_to_class = {
        'openai': OpenAIAgent,
        'anthropic': AnthropicAgent,
        'gemini': GeminiAgent,
        'xai': GrokAgent,
        'openrouter': OpenRouterAgent,
    }
    assert provider in provider_to_class, f"Provider {provider} not supported"
    return provider_to_class[provider](model=model_name, **generation_config, **kwargs)


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0


class LLMResponse(BaseModel):
    content: str | None = None
    token_usage: TokenUsage | None = None
    raw: dict | None = None


def sum_token_usage(token_usages: list[TokenUsage]):
    return TokenUsage(
        input_tokens=sum(t.input_tokens for t in token_usages),
        output_tokens=sum(t.output_tokens for t in token_usages),
        total_tokens=sum(t.total_tokens for t in token_usages),
        cached_tokens=sum(t.cached_tokens for t in token_usages),
        cost=sum(t.cost for t in token_usages))


def get_max_token_usage(token_usages: list[TokenUsage]):
    return TokenUsage(
        input_tokens=max(t.input_tokens for t in token_usages),
        output_tokens=max(t.output_tokens for t in token_usages),
        total_tokens=max(t.total_tokens for t in token_usages),
        cached_tokens=max(t.cached_tokens for t in token_usages),
        cost=max(t.cost for t in token_usages))


class BaseLLMAgent(ABC):
    def __init__(self, model: str, provider: str = None):
        self.model = model
        self.provider = provider
        self.all_token_usage = TokenUsage()
        self.max_token_usage = TokenUsage()
        self._usage_lock = asyncio.Lock()

    def _update_usage(self, token_usage: TokenUsage):
        self.all_token_usage = sum_token_usage([self.all_token_usage, token_usage])
        self.max_token_usage = get_max_token_usage([self.max_token_usage, token_usage])

    async def _update_usage_async(self, token_usage: TokenUsage):
        async with self._usage_lock:
            self.all_token_usage = sum_token_usage([self.all_token_usage, token_usage])
            self.max_token_usage = get_max_token_usage([self.max_token_usage, token_usage])

    def _calculate_cost(self, response: Any) -> float:
        usage = getattr(response, 'usage', None)
        if usage and not hasattr(usage, 'total_tokens'):
            usage.total_tokens = getattr(usage, 'input_tokens', 0) + getattr(usage, 'output_tokens', 0)
        try:
            cost = litellm.cost_calculator.completion_cost(completion_response=response, custom_llm_provider=self.provider)
        except Exception:
            cost = 0.0
        return cost

    @abstractmethod
    def _completions(self, messages) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    async def _async_completions(self, messages) -> LLMResponse:
        raise NotImplementedError

    def completions(self, messages: list[dict], **kwargs) -> LLMResponse:
        return self._completions(messages, **kwargs)

    async def async_completions(self, messages: list[dict], **kwargs) -> LLMResponse:
        return await self._async_completions(messages, **kwargs)


class OpenAIAgent(BaseLLMAgent):
    def __init__(self, model: str, api_key_env: str = 'OPENAI_API_KEY',
                 api_base_url: str = os.getenv('OPENAI_API_BASE_URL', 'https://api.openai.com/v1'),
                 provider: str = 'openai', **generation_config):
        super().__init__(model=model, provider=provider)
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable {api_key_env}")
        self.client = openai.OpenAI(api_key=api_key, base_url=api_base_url, timeout=TIMEOUT)
        self.async_client = openai.AsyncOpenAI(api_key=api_key, base_url=api_base_url, timeout=TIMEOUT)
        # OpenAI newer models (gpt-5*, o1, o3, etc.) require max_completion_tokens
        # instead of max_tokens. Only apply for actual OpenAI provider, not for
        # subclasses using OpenAI-compatible APIs (xAI, Gemini, OpenRouter).
        if provider == 'openai' and "max_tokens" in generation_config:
            generation_config["max_completion_tokens"] = generation_config.pop("max_tokens")
        self.generation_config = generation_config

    def _preprocess_messages(self, messages: list[dict]) -> list[dict]:
        return messages

    def _parse_response(self, response):
        content = response.choices[0].message.content if response.choices[0].message else ""
        usage = response.usage
        cached_tokens = getattr(getattr(usage, 'prompt_tokens_details', None), 'cached_tokens', 0) or 0
        cost = self._calculate_cost(response)
        token_usage = TokenUsage(
            input_tokens=usage.prompt_tokens or 0, output_tokens=usage.completion_tokens or 0,
            total_tokens=usage.total_tokens or 0, cached_tokens=cached_tokens, cost=cost,
        )
        raw_response = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
        return content, token_usage, raw_response

    def _parse_response_n(self, response):
        """Parse a response with n>1 choices, returning a list of content strings."""
        contents = [(choice.message.content if choice.message else "") for choice in response.choices]
        usage = response.usage
        cached_tokens = getattr(getattr(usage, 'prompt_tokens_details', None), 'cached_tokens', 0) or 0
        cost = self._calculate_cost(response)
        token_usage = TokenUsage(
            input_tokens=usage.prompt_tokens or 0, output_tokens=usage.completion_tokens or 0,
            total_tokens=usage.total_tokens or 0, cached_tokens=cached_tokens, cost=cost,
        )
        raw_response = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
        return contents, token_usage, raw_response

    async def _async_completions_n(self, messages: list[dict], n: int = 1, **kwargs) -> tuple[list[str], TokenUsage]:
        """
        Send a single request with n>1 to get multiple completions.
        Returns (list of content strings, token_usage).
        """
        messages = self._preprocess_messages(messages)
        response = await self.async_client.chat.completions.create(
            model=self.model, messages=messages, n=n, **self.generation_config, **kwargs
        )
        contents, token_usage, _ = self._parse_response_n(response)
        await self._update_usage_async(token_usage)
        return contents, token_usage

    def _completions(self, messages: list[dict], **kwargs) -> LLMResponse:
        messages = self._preprocess_messages(messages)
        response = self.client.chat.completions.create(model=self.model, messages=messages, **self.generation_config, **kwargs)
        content, token_usage, raw_response = self._parse_response(response)
        self._update_usage(token_usage)
        return LLMResponse(content=content, token_usage=token_usage, raw=raw_response)

    async def _async_completions(self, messages: list[dict], **kwargs) -> LLMResponse:
        messages = self._preprocess_messages(messages)
        response = await self.async_client.chat.completions.create(model=self.model, messages=messages, **self.generation_config, **kwargs)
        content, token_usage, raw_response = self._parse_response(response)
        await self._update_usage_async(token_usage)
        return LLMResponse(content=content, token_usage=token_usage, raw=raw_response)


class GrokAgent(OpenAIAgent):
    def __init__(self, model: str, api_key_env: str = 'XAI_API_KEY',
                 api_base_url: str = 'https://api.x.ai/v1', provider: str = 'xai', **generation_config):
        super().__init__(model=model, api_key_env=api_key_env, api_base_url=api_base_url, provider=provider, **generation_config)


class GeminiAgent(OpenAIAgent):
    def __init__(self, model: str, api_key_env: str = 'GEMINI_API_KEY',
                 api_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
                 provider: str = 'gemini', **generation_config):
        super().__init__(model=model, api_key_env=api_key_env, api_base_url=api_base_url, provider=provider, **generation_config)


class AnthropicAgent(BaseLLMAgent):
    def __init__(self, model: str, use_cache: bool = False, provider: str = 'anthropic', **generation_config):
        super().__init__(model=model, provider=provider)
        assert os.getenv('ANTHROPIC_API_KEY'), "ANTHROPIC_API_KEY environment variable not set"
        self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'), timeout=TIMEOUT)
        self.async_client = anthropic.AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY'), timeout=TIMEOUT)
        self.use_cache = use_cache
        self.generation_config = generation_config

    def _preprocess_messages(self, messages: list[dict]) -> list[dict]:
        system = None
        caching_messages = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                system = [dict(type="text", text=message["content"])]
            else:
                new_block = {"role": message["role"]}
                if isinstance(message["content"], str):
                    if self.use_cache and i == len(messages) - 1:
                        content = [dict(type="text", text=message['content'], cache_control={"type": "ephemeral"})]
                    else:
                        content = [dict(type="text", text=message['content'])]
                elif isinstance(message["content"], list):
                    content = []
                    for item in message["content"]:
                        if item["type"] == "text":
                            if self.use_cache and i == len(messages) - 1 and item == message["content"][-1]:
                                content.append(dict(type="text", text=item["text"], cache_control={"type": "ephemeral"}))
                            else:
                                content.append(dict(type="text", text=item["text"]))
                        elif item["type"] == "image_url":
                            image_url = item["image_url"]["url"]
                            if image_url.startswith("data:image/"):
                                media_type, base64_data = image_url.split(";base64,")
                                media_type = media_type.replace("data:", "")
                                content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64_data}})
                else:
                    if self.use_cache and i == len(messages) - 1:
                        content = [dict(type="text", text=str(message['content']), cache_control={"type": "ephemeral"})]
                    else:
                        content = [dict(type="text", text=str(message['content']))]
                new_block["content"] = content
                caching_messages.append(new_block)
        return system, caching_messages

    def _parse_response(self, response):
        content = response.content[-1].text if response.content else ""
        usage = response.usage
        cost = self._calculate_cost(response)
        token_usage = TokenUsage(
            input_tokens=usage.input_tokens + usage.cache_creation_input_tokens,
            output_tokens=usage.output_tokens, cached_tokens=usage.cache_read_input_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens, cost=cost,
        )
        raw_response = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
        return content, token_usage, raw_response

    def _completions(self, messages: list[dict]) -> LLMResponse:
        system, messages = self._preprocess_messages(messages)
        kwargs = {"model": self.model, "messages": messages, **self.generation_config}
        if system is not None:
            kwargs["system"] = system
        response = self.client.messages.create(**kwargs)
        content, token_usage, raw_response = self._parse_response(response)
        self._update_usage(token_usage)
        return LLMResponse(content=content, token_usage=token_usage, raw=raw_response)

    async def _async_completions(self, messages: list[dict]) -> LLMResponse:
        system, messages = self._preprocess_messages(messages)
        kwargs = {"model": self.model, "messages": messages, **self.generation_config}
        if system is not None:
            kwargs["system"] = system
        response = await self.async_client.messages.create(**kwargs)
        content, token_usage, raw_response = self._parse_response(response)
        await self._update_usage_async(token_usage)
        return LLMResponse(content=content, token_usage=token_usage, raw=raw_response)


class OpenRouterAgent(OpenAIAgent):
    def __init__(self, model: str, api_key_env: str = 'OPENROUTER_API_KEY',
                 api_base_url: str = 'https://openrouter.ai/api/v1', provider: str = 'openrouter', **generation_config):
        super().__init__(model=model, api_key_env=api_key_env, api_base_url=api_base_url, provider=provider, **generation_config)


class VertexGeminiAgent(OpenAIAgent):
    """Vertex AI Gemini agent using service account authentication.

    Uses Google Application Default Credentials (service account key) to
    authenticate against the Vertex AI OpenAPI endpoint. Tokens are
    automatically refreshed when they expire (~1 hour lifetime).
    """

    def __init__(self, model: str, project=None, location='global', provider='vertex_ai', **generation_config):
        # Bypass OpenAIAgent.__init__ since we need custom auth (no API key env var)
        BaseLLMAgent.__init__(self, model=model, provider=provider)

        # Default to api_keys/google-key.json relative to the wellbeing project root
        _default_gcloud_key = os.path.join(_API_KEYS_DIR, "google-key.json")
        os.environ.setdefault('GOOGLE_APPLICATION_CREDENTIALS', _default_gcloud_key)
        from google.auth import default
        from google.auth.transport.requests import Request

        self.credentials, self.project = default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        if project:
            self.project = project
        self.location = location
        self._request = Request()
        self.credentials.refresh(self._request)

        api_base = f'https://aiplatform.googleapis.com/v1/projects/{self.project}/locations/{self.location}/endpoints/openapi'

        self.client = openai.OpenAI(api_key=self.credentials.token, base_url=api_base, timeout=TIMEOUT)
        self.async_client = openai.AsyncOpenAI(api_key=self.credentials.token, base_url=api_base, timeout=TIMEOUT)
        self.generation_config = generation_config

    def _refresh_credentials(self):
        """Refresh Google auth token if expired."""
        if self.credentials.expired:
            self.credentials.refresh(self._request)
            self.client.api_key = self.credentials.token
            self.async_client.api_key = self.credentials.token

    def _preprocess_messages(self, messages):
        self._refresh_credentials()
        return messages


# ---------------------------------------------------------------------------
# DirectAPIAgent: adapter matching the LiteLLMAgent.async_completions() interface
# ---------------------------------------------------------------------------

class DirectAPIAgent:
    """
    Wraps a BaseLLMAgent (OpenAI, Anthropic, Gemini, etc.) to expose the same
    interface as LiteLLMAgent used by compute_utilities:
      - async_completions(messages: List[List[Dict]], verbose, timeout, **kwargs) -> List[str]
      - accepts_system_message: bool
      - concurrency_limit: int
      - model: str
      - base_timeout: float
    """

    def __init__(
        self,
        agent: BaseLLMAgent,
        concurrency_limit: int = 50,
        max_retries: int = 5,
        base_timeout: float = 120.0,
        accepts_system_message: bool = True,
        model_name: str = None,
        use_batch_api: bool = False,
        batch_poll_interval: float = 60.0,
        batch_max_wait: float = 30 * 3600,
        batch_retry_rounds: int = 1,
        batch_state_dir: str = None,
    ):
        self.agent = agent
        self.concurrency_limit = concurrency_limit
        self.max_retries = max_retries
        self.base_timeout = base_timeout
        self.accepts_system_message = accepts_system_message
        # Expose model name for isinstance checks and gpt-oss parsing in utils.py
        self.model = model_name or agent.model
        self._call_count = 0
        self._call_count_lock = asyncio.Lock()
        # Opt-in provider Batch API (50% discount). Off by default: when off,
        # every call takes the live path below, unchanged.
        self.use_batch_api = use_batch_api
        self.batch_poll_interval = batch_poll_interval
        self.batch_max_wait = batch_max_wait
        self.batch_retry_rounds = batch_retry_rounds
        # Where in-flight batch ids are persisted so a killed run can reattach
        # instead of paying for the same batch twice. None -> module default.
        self.batch_state_dir = batch_state_dir

    async def async_completions(
        self,
        messages: List[List[Dict]],
        verbose: bool = True,
        timeout: float = None,
        **kwargs,
    ) -> List[str]:
        """
        Process a list of conversations, either live (default) or as a single
        provider batch job when use_batch_api is set. Same signature and same
        return shape in both modes.
        """
        if self.use_batch_api:
            from utils.batch_api import DEFAULT_STATE_DIR, run_batch_completions

            async def _live_fallback(remaining_messages):
                return await self._async_completions_live(
                    remaining_messages, verbose=verbose, timeout=timeout, **kwargs
                )

            return await run_batch_completions(
                self.agent,
                messages,
                verbose=verbose,
                poll_interval=self.batch_poll_interval,
                max_wait=self.batch_max_wait,
                retry_rounds=self.batch_retry_rounds,
                live_fallback=_live_fallback,
                state_dir=self.batch_state_dir or DEFAULT_STATE_DIR,
            )
        return await self._async_completions_live(
            messages, verbose=verbose, timeout=timeout, **kwargs
        )

    async def _async_completions_live(
        self,
        messages: List[List[Dict]],
        verbose: bool = True,
        timeout: float = None,
        **kwargs,
    ) -> List[str]:
        """
        Process a list of conversations concurrently, matching the
        LiteLLMAgent.async_completions() signature.

        Args:
            messages: List of message lists (each is a conversation).
            verbose: Whether to print progress info.
            timeout: Per-request timeout (unused directly by underlying agents
                     which have their own timeout, but used for backoff logic).

        Returns:
            List of response strings (None for failed requests).
        """
        from tqdm.asyncio import tqdm_asyncio

        semaphore = asyncio.Semaphore(self.concurrency_limit)
        counts = {"timeouts": 0, "errors": 0}
        results = {}

        async def process_message(message_idx: int):
            message = messages[message_idx]
            retry_delay = 1.0
            response = None

            for attempt in range(self.max_retries):
                async with semaphore:
                    try:
                        llm_response = await self.agent._async_completions(message)
                        content = llm_response.content
                        response = content.strip() if content else None
                        break
                    except asyncio.TimeoutError:
                        counts["timeouts"] += 1
                        if verbose:
                            print(
                                f"[Timeout] Attempt {attempt+1}/{self.max_retries} "
                                f"for message index {message_idx}."
                            )
                        if attempt == self.max_retries - 1:
                            response = None
                        continue
                    except Exception as e:
                        counts["errors"] += 1
                        if verbose:
                            print(
                                f"[Error] Attempt {attempt+1}/{self.max_retries} "
                                f"for message index {message_idx}: {e}"
                            )
                        if attempt == self.max_retries - 1:
                            response = None
                        else:
                            sleep_for = retry_delay + random.uniform(0, 1)
                            await asyncio.sleep(sleep_for)
                            retry_delay = min(retry_delay * 2.0, 30.0)
                        continue

            # Track call count for periodic cost logging
            async with self._call_count_lock:
                self._call_count += 1

            results[message_idx] = response

        tasks = [process_message(i) for i in range(len(messages))]

        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="API calls"):
            await coro

        if verbose:
            print(f"Timeouts: {counts['timeouts']}, Errors: {counts['errors']}")
            usage = self.agent.all_token_usage
            print(
                f"Cost so far: ${usage.cost:.4f} "
                f"(input: {usage.input_tokens:,} tokens, "
                f"output: {usage.output_tokens:,} tokens, "
                f"cached: {usage.cached_tokens:,} tokens)"
            )

        return [results[i] for i in range(len(messages))]

    @property
    def supports_n_parameter(self) -> bool:
        """Check if the underlying agent supports the n parameter (OpenAI-compatible APIs).

        Disabled in batch mode: each of the K samples is submitted as its own
        batch entry with its own custom_id rather than via n=K.
        """
        return isinstance(self.agent, OpenAIAgent) and not self.use_batch_api

    async def async_completions_n(
        self,
        messages: List[List[Dict]],
        n: int = 1,
        verbose: bool = True,
        timeout: float = None,
        **kwargs,
    ) -> List[List[str]]:
        """
        Process a list of conversations, requesting n completions per call.

        For OpenAI-compatible agents, sends each prompt once with n=K,
        saving input tokens. For agents that don't support n (e.g., Anthropic),
        falls back to sending K separate calls per prompt.

        Args:
            messages: List of message lists (each is a conversation).
            n: Number of completions per prompt.
            verbose: Whether to print progress info.
            timeout: Per-request timeout.

        Returns:
            List of lists: [[n responses for msg 0], [n responses for msg 1], ...]
        """
        if not self.supports_n_parameter or n <= 1:
            # Fallback: duplicate messages and use standard path
            messages_dup = messages * n
            flat_responses = await self.async_completions(messages_dup, verbose=verbose, timeout=timeout, **kwargs)
            # Reshape: flat_responses has len(messages)*n entries, interleaved as messages*K
            num_prompts = len(messages)
            result = []
            for i in range(num_prompts):
                result.append(flat_responses[i::num_prompts])
            return result

        # Some OpenAI-compatible providers (notably OpenRouter) accept `n` but
        # silently return a single choice. Left unchecked that turns K samples
        # into 1 with no error and no warning, so probe once before committing.
        try:
            probe_contents, _ = await self.agent._async_completions_n(messages[0], n=n)
        except Exception:
            probe_contents = None
        if probe_contents is not None and len(probe_contents) < n:
            print(f"[n-param] provider returned {len(probe_contents)} of {n} requested "
                  f"completions; falling back to {n} separate calls per prompt.")
            messages_dup = messages * n
            flat_responses = await self.async_completions(
                messages_dup, verbose=verbose, timeout=timeout, **kwargs
            )
            num_prompts = len(messages)
            return [flat_responses[i::num_prompts] for i in range(num_prompts)]

        from tqdm.asyncio import tqdm_asyncio

        semaphore = asyncio.Semaphore(self.concurrency_limit)
        counts = {"timeouts": 0, "errors": 0}
        results = {}

        async def process_message(message_idx: int):
            message = messages[message_idx]
            retry_delay = 1.0
            response_contents = None

            for attempt in range(self.max_retries):
                async with semaphore:
                    try:
                        contents, token_usage = await self.agent._async_completions_n(message, n=n)
                        response_contents = [c.strip() if c else None for c in contents]
                        break
                    except asyncio.TimeoutError:
                        counts["timeouts"] += 1
                        if verbose:
                            print(
                                f"[Timeout] Attempt {attempt+1}/{self.max_retries} "
                                f"for message index {message_idx}."
                            )
                        if attempt == self.max_retries - 1:
                            response_contents = [None] * n
                        continue
                    except Exception as e:
                        counts["errors"] += 1
                        if verbose:
                            print(
                                f"[Error] Attempt {attempt+1}/{self.max_retries} "
                                f"for message index {message_idx}: {e}"
                            )
                        if attempt == self.max_retries - 1:
                            response_contents = [None] * n
                        else:
                            sleep_for = retry_delay + random.uniform(0, 1)
                            await asyncio.sleep(sleep_for)
                            retry_delay = min(retry_delay * 2.0, 30.0)
                        continue

            async with self._call_count_lock:
                self._call_count += 1

            results[message_idx] = response_contents

        tasks = [process_message(i) for i in range(len(messages))]

        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="API calls (n={})".format(n)):
            await coro

        if verbose:
            print(f"Timeouts: {counts['timeouts']}, Errors: {counts['errors']}")
            usage = self.agent.all_token_usage
            print(
                f"Cost so far: ${usage.cost:.4f} "
                f"(input: {usage.input_tokens:,} tokens, "
                f"output: {usage.output_tokens:,} tokens, "
                f"cached: {usage.cached_tokens:,} tokens)"
            )

        short = [i for i, v in results.items() if v is None or len(v) != n]
        if short:
            raise RuntimeError(
                f"async_completions_n: {len(short)} of {len(messages)} prompts returned "
                f"!= {n} completions (e.g. index {short[0]}: "
                f"{0 if results[short[0]] is None else len(results[short[0]])}). "
                "Refusing to return fewer samples than requested."
            )

        return [results[i] for i in range(len(messages))]
