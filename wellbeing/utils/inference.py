"""
Unified inference wrapper for vLLM (local) and API-based model generation.

For utility computation (pairwise preferences with active learning), use
compute_utilities/ instead.

Usage:
    from utils.inference import load_vllm_engine, generate_vllm, generate_api, generate

    # vLLM (caller manages engine lifetime)
    llm, tokenizer = load_vllm_engine("qwen25-7b-instruct")
    results = generate_vllm(llm, tokenizer, messages_list, n=5, temperature=1.0)

    # API
    results = generate_api("gpt-4o", messages_list, n=1, temperature=0.0)

    # Auto-dispatch
    results = generate("qwen25-7b-instruct", messages_list, llm=llm, tokenizer=tokenizer)
"""
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_utils import get_model_config, get_model_type

logger = logging.getLogger(__name__)

# Global vLLM engine cache (keyed by model_key)
_vllm_engines: Dict[str, Tuple[Any, Any]] = {}  # model_key -> (llm, tokenizer)

API_MODEL_TYPES = frozenset([
    "openai", "anthropic", "gdm", "xai", "togetherai", "litellm_proxy",
])

# Direct API model types -- these use api_agents.py instead of litellm
DIRECT_API_MODEL_TYPES = frozenset([
    "openai_direct", "anthropic_direct", "gemini_direct", "xai_direct", "openrouter_direct",
    "vertex_gemini_direct",
])

# All API model types (union of litellm and direct)
ALL_API_MODEL_TYPES = API_MODEL_TYPES | DIRECT_API_MODEL_TYPES


def is_api_model(
    model_key: str,
    models_config_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Check if a model_key refers to an API model (not local vLLM)."""
    cfg = get_model_config(model_key, models_config_path=models_config_path)
    return cfg.get("model_type", "vllm") in ALL_API_MODEL_TYPES


# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------

def load_vllm_engine(
    model_key: str,
    models_config_path: Optional[str] = None,
    cache: bool = True,
    **extra_llm_kwargs,
) -> Tuple[Any, Any]:
    """Load a vLLM engine and tokenizer for a model.

    Args:
        model_key: Key from models.yaml.
        models_config_path: Optional override for models.yaml path.
        cache: If True, reuse previously loaded engines for the same model_key.
        **extra_llm_kwargs: Extra kwargs passed to vllm.LLM (e.g. max_model_len,
            gpu_memory_utilization, limit_mm_per_prompt, enforce_eager).

    Returns:
        (llm, tokenizer) tuple.
    """
    if cache and model_key in _vllm_engines:
        logger.info("Reusing cached vLLM engine for %s", model_key)
        return _vllm_engines[model_key]

    from vllm import LLM

    if models_config_path:
        import yaml
        with open(models_config_path, "r") as f:
            models = yaml.safe_load(f)
        config = models[model_key]
    else:
        config = get_model_config(model_key)

    model_path = config.get("path", config["model_name"])
    tp_size = config.get("gpu_count", 1)

    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
    )
    # Allow env var override for max_model_len
    env_max_model_len = os.environ.get("VLLM_MAX_MODEL_LEN")
    if env_max_model_len and "max_model_len" not in extra_llm_kwargs:
        llm_kwargs["max_model_len"] = int(env_max_model_len)
    # Pass vllm_kwargs from model config (gpu_memory_utilization, enforce_eager, etc.)
    vllm_kwargs = config.get("vllm_kwargs", {})
    for k, v in vllm_kwargs.items():
        if k not in extra_llm_kwargs:
            llm_kwargs[k] = v
    # Apply any extra kwargs (caller can override defaults)
    llm_kwargs.update(extra_llm_kwargs)

    logger.info("Loading vLLM engine: %s (TP=%d) ...", model_path, tp_size)
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    # Some models (e.g. Qwen3-Omni) don't set chat_template on the tokenizer
    # but provide it in a separate chat_template.json file.
    if not getattr(tokenizer, 'chat_template', None):
        import json as _json
        ct_path = os.path.join(model_path, "chat_template.json")
        if os.path.exists(ct_path):
            with open(ct_path) as _f:
                tokenizer.chat_template = _json.load(_f)["chat_template"]
            logger.info("Loaded chat_template from %s", ct_path)

    if cache:
        _vllm_engines[model_key] = (llm, tokenizer)

    return llm, tokenizer


def setup_api_client(
    model_key: str,
    models_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load API configuration for a model.

    Returns a dict with keys: model_name, model_type, api_key (optional),
    base_url (optional), max_tokens (optional), timeout (optional),
    no_temperature (bool).
    Sets the appropriate environment variable for the API key.
    """
    if models_config_path:
        import yaml
        with open(models_config_path, "r") as f:
            models = yaml.safe_load(f)
        config = models[model_key]
    else:
        config = get_model_config(model_key)

    model_type = config["model_type"]
    model_name = config["model_name"]

    result = {
        "model_name": model_name,
        "model_type": model_type,
        "max_tokens": config.get("max_tokens"),
        # Generation-only ceiling; see generate_api_direct for why it is separate.
        "generation_max_tokens": config.get("generation_max_tokens"),
        "timeout": config.get("timeout", 30),
        # Models that reject the temperature parameter (HTTP 400 if sent).
        "no_temperature": config.get("no_temperature", False),
    }

    if model_type == "litellm_proxy":
        # Load proxy API key and base_url
        api_key = config.get("api_key")
        if not api_key:
            api_key_path = PROJECT_ROOT / "api_keys" / "api_key_litellm_proxy.txt"
            try:
                api_key = api_key_path.read_text().strip()
            except FileNotFoundError:
                raise ValueError(f"No API key file found at {api_key_path}")
        result["api_key"] = api_key
        result["base_url"] = config.get("base_url")
        # litellm needs "openai/" prefix for OpenAI-compatible routing
        if not model_name.startswith("openai/"):
            result["model_name"] = f"openai/{model_name}"
    else:
        # Standard API providers: set env var from key file
        api_key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gdm": "GEMINI_API_KEY",
            "xai": "XAI_API_KEY",
            "togetherai": "TOGETHER_AI_API_KEY",
        }
        if model_type in api_key_map:
            env_var = api_key_map[model_type]
            if env_var not in os.environ or not os.environ[env_var]:
                api_key_path = PROJECT_ROOT / "api_keys" / f"api_key_{model_type}.txt"
                try:
                    api_key = api_key_path.read_text().strip()
                except FileNotFoundError:
                    raise ValueError(
                        f"No API key file at {api_key_path} and {env_var} not set."
                    )
                if not api_key:
                    raise ValueError(f"API key file {api_key_path} is empty.")
                os.environ[env_var] = api_key

    return result


# ---------------------------------------------------------------------------
#  vLLM generation
# ---------------------------------------------------------------------------

def generate_vllm(
    llm,
    tokenizer,
    messages_list: List[List[Dict[str, str]]],
    *,
    n: int = 1,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    top_p: float = 1.0,
    stop: Optional[List[str]] = None,
    chat_template_kwargs: Optional[Dict] = None,
    seed: Optional[int] = None,
) -> List[List[str]]:
    """Batch generate with a vLLM engine.

    Args:
        llm: vLLM LLM instance.
        tokenizer: Tokenizer (from llm.get_tokenizer() or AutoTokenizer).
        messages_list: List of chat-format message lists, each like
            [{"role": "user", "content": "..."}].
        n: Number of completions per prompt.
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate.
        top_p: Nucleus sampling parameter.
        stop: Optional stop sequences.
        chat_template_kwargs: Extra kwargs for tokenizer.apply_chat_template().
        seed: Optional sampling seed for reproducible generation.

    Returns:
        List of lists of strings. Outer list has len(messages_list) entries,
        inner lists have n entries each.
    """
    from vllm import SamplingParams

    ct_kwargs = chat_template_kwargs or {}

    # Apply chat template to convert messages -> prompt strings
    prompts = []
    for messages in messages_list:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **ct_kwargs,
        )
        prompts.append(prompt_text)

    sp_kwargs = dict(
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
    )
    if seed is not None:
        sp_kwargs["seed"] = seed
    sampling_params = SamplingParams(**sp_kwargs)

    outputs = llm.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        completions = [c.text for c in output.outputs]
        results.append(completions)

    return results


# ---------------------------------------------------------------------------
#  API generation (via LiteLLM)
# ---------------------------------------------------------------------------

async def generate_api(
    model_key_or_config: Union[str, Dict],
    messages_list: List[List[Dict[str, str]]],
    *,
    n: int = 1,
    temperature: Optional[float] = None,
    max_tokens: int = 1024,
    concurrency: int = 10,
    models_config_path: Optional[str] = None,
) -> List[List[str]]:
    """Async batch generate via LiteLLM.

    Args:
        model_key_or_config: Either a model key string (looked up in models.yaml)
            or a pre-built config dict from setup_api_client().
        messages_list: List of chat-format message lists.
        n: Number of completions per prompt.
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate.
        concurrency: Max concurrent API requests.
        models_config_path: Optional override for models.yaml path.

    Returns:
        List of lists of strings. Outer list has len(messages_list) entries,
        inner lists have n entries each.
    """
    import litellm

    if isinstance(model_key_or_config, str):
        api_config = setup_api_client(model_key_or_config, models_config_path)
    else:
        api_config = model_key_or_config

    model_name = api_config["model_name"]
    # `generation_max_tokens` is a generation-only ceiling. Reasoning models bill
    # hidden reasoning tokens against the same budget as visible content, so a
    # limit that is fine for one-token A/B comparisons can leave a long-form
    # answer with zero content (finish_reason="length", empty string). Raising
    # `max_tokens` instead would also raise the comparison budget, so keep them
    # separate. A ceiling is free unless it is actually used.
    api_max_tokens = max(
        max_tokens,
        api_config.get("generation_max_tokens") or api_config.get("max_tokens") or 0,
    )
    api_timeout = api_config.get("timeout", 30)

    # Some models reject the temperature parameter outright (HTTP 400). Models
    # can opt out via `no_temperature: true` in models.yaml.
    if api_config.get("no_temperature"):
        temperature = None

    # Build extra kwargs for litellm
    extra_kwargs = {}
    if api_config.get("api_key"):
        extra_kwargs["api_key"] = api_config["api_key"]
    if api_config.get("base_url"):
        extra_kwargs["base_url"] = api_config["base_url"]

    semaphore = asyncio.Semaphore(concurrency)

    max_retries = 5

    async def _call(messages: List[Dict]) -> str:
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    call_kwargs = dict(
                        model=model_name,
                        messages=messages,
                        max_tokens=api_max_tokens,
                        timeout=api_timeout,
                        **extra_kwargs,
                    )
                    if temperature is not None:
                        call_kwargs["temperature"] = temperature
                    response = await litellm.acompletion(**call_kwargs)
                    return response.choices[0].message.content or ""
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning("API call failed (attempt %d/%d), retrying in %ds: %s",
                                       attempt + 1, max_retries, wait, e)
                        await asyncio.sleep(wait)
                    else:
                        logger.warning("API call failed after %d attempts: %s", max_retries, e)
                        return ""

    # Build all tasks: for each prompt, generate n completions
    all_tasks = []
    task_map = []  # (prompt_idx, sample_idx)
    for prompt_idx, messages in enumerate(messages_list):
        for sample_idx in range(n):
            all_tasks.append(_call(messages))
            task_map.append((prompt_idx, sample_idx))

    raw_responses = await asyncio.gather(*all_tasks)

    # Organize into list[list[str]]
    results: List[List[str]] = [[] for _ in range(len(messages_list))]
    for (prompt_idx, _), response_text in zip(task_map, raw_responses):
        results[prompt_idx].append(response_text)

    return results


# ---------------------------------------------------------------------------
#  Direct API generation (using api_agents.py)
# ---------------------------------------------------------------------------

async def generate_api_direct(
    model_key: str,
    messages_list: List[List[Dict[str, str]]],
    *,
    n: int = 1,
    temperature: Optional[float] = None,
    max_tokens: int = 1024,
    concurrency: int = 50,
    models_config_path: Optional[str] = None,
) -> List[List[str]]:
    """Async batch generate via direct API agents (api_agents.py).

    Used for model types ending in '_direct' (openai_direct, anthropic_direct, etc.).
    These bypass litellm and call provider APIs directly.

    Args:
        model_key: Model key from models.yaml.
        messages_list: List of chat-format message lists.
        n: Number of completions per prompt.
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate.
        concurrency: Max concurrent API requests.
        models_config_path: Optional override for models.yaml path.

    Returns:
        List of lists of strings. Outer list has len(messages_list) entries,
        inner lists have n entries each.
    """
    from utils.api_agents import (
        DirectAPIAgent,
        OpenAIAgent as DirectOpenAIAgent,
        AnthropicAgent as DirectAnthropicAgent,
        GeminiAgent as DirectGeminiAgent,
        GrokAgent as DirectGrokAgent,
        OpenRouterAgent as DirectOpenRouterAgent,
        VertexGeminiAgent as DirectVertexGeminiAgent,
    )

    if models_config_path:
        import yaml
        with open(models_config_path, "r") as f:
            models = yaml.safe_load(f)
        config = models[model_key]
    else:
        config = get_model_config(model_key)

    model_type = config["model_type"]
    model_name = config["model_name"]

    provider_map = {
        "openai_direct": DirectOpenAIAgent,
        "anthropic_direct": DirectAnthropicAgent,
        "gemini_direct": DirectGeminiAgent,
        "xai_direct": DirectGrokAgent,
        "openrouter_direct": DirectOpenRouterAgent,
        "vertex_gemini_direct": DirectVertexGeminiAgent,
    }
    agent_cls = provider_map[model_type]

    # Build generation config
    generation_config = {}
    # Some models reject the temperature parameter outright (HTTP 400). Models
    # can opt out via `no_temperature: true` in models.yaml.
    if config.get("no_temperature"):
        temperature = None
    if temperature is not None:
        generation_config["temperature"] = temperature
    # Honour the generation-only ceiling here as well as in generate_api(); the
    # agent path is what *_direct model types actually use.
    generation_config["max_tokens"] = max(
        max_tokens, config.get("generation_max_tokens") or 0
    )
    if "reasoning_effort" in config:
        generation_config["reasoning_effort"] = config["reasoning_effort"]

    # For Anthropic, max_tokens is required
    if model_type == "anthropic_direct" and "max_tokens" not in generation_config:
        generation_config["max_tokens"] = max_tokens

    base_timeout = config.get("timeout", 120.0)

    # Pass custom API base URL and key env var if specified in model config
    agent_kwargs = {}
    if "api_base_url" in config:
        agent_kwargs["api_base_url"] = config["api_base_url"]
        # Custom base URL = not actual OpenAI; avoid max_completion_tokens renaming
        agent_kwargs["provider"] = "openai_compatible"
    if "api_key_env" in config:
        agent_kwargs["api_key_env"] = config["api_key_env"]

    underlying_agent = agent_cls(model=model_name, **agent_kwargs, **generation_config)

    agent = DirectAPIAgent(
        agent=underlying_agent,
        concurrency_limit=concurrency,
        max_retries=5,
        base_timeout=base_timeout,
        accepts_system_message=True,
        model_name=model_name,
        # Opt-in provider Batch API (see utils/batch_api.py); default off.
        use_batch_api=config.get("use_batch_api", False),
        batch_poll_interval=config.get("batch_poll_interval", 60.0),
        batch_max_wait=config.get("batch_max_wait", 30 * 3600),
        batch_retry_rounds=config.get("batch_retry_rounds", 1),
        batch_state_dir=config.get("batch_state_dir"),
    )

    # For n > 1 samples, repeat each message n times and re-group
    if n == 1:
        raw_responses = await agent.async_completions(messages_list, verbose=True)
        return [[r or ""] for r in raw_responses]
    else:
        expanded_messages = []
        task_map = []
        for prompt_idx, messages in enumerate(messages_list):
            for sample_idx in range(n):
                expanded_messages.append(messages)
                task_map.append((prompt_idx, sample_idx))

        raw_responses = await agent.async_completions(expanded_messages, verbose=True)

        results: List[List[str]] = [[] for _ in range(len(messages_list))]
        for (prompt_idx, _), response_text in zip(task_map, raw_responses):
            results[prompt_idx].append(response_text or "")

        return results


# ---------------------------------------------------------------------------
#  Unified entry point
# ---------------------------------------------------------------------------

def generate(
    model_key: str,
    messages_list: List[List[Dict[str, str]]],
    *,
    n: int = 1,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    top_p: float = 1.0,
    stop: Optional[List[str]] = None,
    llm=None,
    tokenizer=None,
    models_config_path: Optional[str] = None,
    concurrency: int = 10,
    chat_template_kwargs: Optional[Dict] = None,
) -> List[List[str]]:
    """Auto-dispatch to vLLM or API based on model_type.

    If llm/tokenizer are provided, uses them directly (avoids reloading).
    For API models, runs the async generate_api in an event loop.

    Args:
        model_key: Key from models.yaml.
        messages_list: List of chat-format message lists.
        n: Number of completions per prompt.
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate.
        top_p: Nucleus sampling.
        stop: Stop sequences.
        llm: Pre-loaded vLLM LLM instance (optional, for vLLM models).
        tokenizer: Pre-loaded tokenizer (optional, for vLLM models).
        models_config_path: Optional override for models.yaml path.
        concurrency: Max concurrent API requests (API models only).
        chat_template_kwargs: Extra kwargs for chat template (vLLM only).

    Returns:
        List of lists of strings. Outer list has len(messages_list) entries,
        inner lists have n entries each.
    """
    if models_config_path:
        import yaml
        with open(models_config_path, "r") as f:
            models = yaml.safe_load(f)
        config = models[model_key]
        model_type = config["model_type"]
    else:
        model_type = get_model_type(model_key)

    if model_type in ("vllm", "vllm_base_model"):
        if llm is None or tokenizer is None:
            llm, tokenizer = load_vllm_engine(
                model_key, models_config_path=models_config_path,
            )
        return generate_vllm(
            llm, tokenizer, messages_list,
            n=n, temperature=temperature, max_tokens=max_tokens,
            top_p=top_p, stop=stop, chat_template_kwargs=chat_template_kwargs,
        )
    elif model_type in API_MODEL_TYPES:
        return asyncio.run(
            generate_api(
                model_key, messages_list,
                n=n, temperature=temperature, max_tokens=max_tokens,
                concurrency=concurrency, models_config_path=models_config_path,
            )
        )
    elif model_type in DIRECT_API_MODEL_TYPES:
        return asyncio.run(
            generate_api_direct(
                model_key, messages_list,
                n=n, temperature=temperature, max_tokens=max_tokens,
                concurrency=concurrency, models_config_path=models_config_path,
            )
        )
    else:
        raise ValueError(f"Unsupported model type '{model_type}' for model '{model_key}'")
