"""
# models/model_manager.py

Module Contract
- Purpose: Single interface over local HF models and API chat models (OpenRouter/OpenAI). Handles registration, switching, sync/async generation, and embedding model access.
- Inputs:
  - generate(prompt, model_name=?, system_prompt=?, …)
  - generate_async(prompt, raw=?, system_prompt=?)
  - generate_once(prompt, model_name=?, system_prompt=?, max_tokens=?)
  - generate_once_with_tools(prompt, model_name=?, tools=?, tool_choice=?)
- Outputs:
  - Text responses (sync) or async stream of ChatCompletion chunks; stub output when API unavailable.
  - For generate_once_with_tools: raw response object with tool_calls for parsing
- Key methods:
  - load_model(), load_openai_model(), switch_model(), get_active_model_name(), get_embedder()
  - truncate_prompt(): ensures local prompts fit context window
  - Model capability registry [NEW 2026-07-22]: module-level MODEL_CAPABILITIES is the SINGLE
    SOURCE OF TRUTH (keyed by full OpenRouter slug: reasoning/vision/tools/caching), alongside
    API_MODEL_ALIASES. The four supports_* methods delegate to pure _slug_supports_*() functions
    derived from it. Registering a model without a caps row — or a classifier drifting from the
    declared intent — fails loudly (tests/unit/test_model_capability_wiring.py); the deployed
    table is validated against OpenRouter's live /models metadata by
    scripts/verify_model_capabilities_live.py. This closed the recurring "registered but silently
    feature-disabled" dead-wiring class (e.g. Kimi K3 was tool-disabled; fable-5 vision+tools OFF).
    [2026-07-30] Rows may also declare "forced_top_p": a top_p value the endpoint MANDATES
    (Kimi K3 400s on anything but 0.95). resolve_top_p(model, requested) is applied on every
    API path that sends top_p (generate_with_openai / generate_once / generate_async) and
    overrides caller-supplied values for such models; tests/unit/test_forced_top_p.py.
  - supports_tools(): checks if a model supports function/tool calling
  - supports_vision(model_name): image/vision input (GPT-4o+, ALL Claude incl. fable-5, Gemini,
    Kimi K3/K2.6; False for DeepSeek, GLM). generate_async() now requests reasoning separation even
    on image turns — multimodal reasoning models support both [UPDATED 2026-07-22]
  - supports_reasoning(model_name): extended thinking/reasoning (Anthropic Claude, DeepSeek R1+v4,
    Moonshot Kimi K3/K2-Thinking/K2.6) [UPDATED 2026-07-22]
  - generate_once(): handles list-of-content-blocks responses (Anthropic extended thinking) by extracting text blocks [ENHANCED 2026-03-26]
  - generate_async() and generate_once(): pass extra_body={"reasoning": {"effort": "medium"}} for
    reasoning models so thinking arrives via delta.reasoning_content (API-level separation) [ENHANCED 2026-04-05]
  - generate_async(disable_reasoning=True) / generate_once(disable_reasoning=True): suppress that
    reasoning separation even for reasoning-capable models. Used as a recovery retry when a model
    (e.g. deepseek-v4) swallows its whole answer into the reasoning channel and returns empty
    visible content — disabling reasoning forces the answer into normal content [NEW 2026-06-14]
  - Prompt caching (OpenRouter) [NEW 2026-06-26]: _format_messages_with_cache() splits the system
    prompt at PROMPT_CACHE_BREAKPOINT — the stable prefix (personality+principles+identity, inserted
    by the orchestrator) carries cache_control:ephemeral; the per-turn volatile tail follows as a
    second, uncached system block. _strip_cache_breakpoint() removes the marker on every non-cache
    path so it never reaches the model. supports_prompt_caching() gates this to Anthropic / recent
    GPT models (the active deepseek-v4 auto-caches server-side, no cache_control). All API paths
    request OpenRouter usage accounting (extra_body usage.include) + stream_options.include_usage,
    and call _log_cache_usage() to emit greppable [PromptCache] HIT|WRITE|MISS lines.
- Dependencies:
  - transformers, sentence-transformers, httpx, environment OPENAI_API_KEY
- Side effects:
  - Maintains HTTP clients; exposes aclose/close to release resources.

Additional Contract (Tool Calling):
  - generate_once_with_tools() supports function/tool calling for agentic workflows
  - Returns raw response with tool_calls attribute for parsing
  - Gracefully handles models without tool support (falls back to standard generation)
  - supports_tools(model_name) returns bool indicating capability
"""
# Import dependencies and config defaults
from utils.logging_utils import log_and_time, get_logger
# Use the root logger or create a child logger that will inherit handlers
logger = get_logger("main")
import os
from config.app_config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K, SYSTEM_PROMPT, PROMPT_CACHE_BREAKPOINT
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import time
import torch
# Optional OpenAI dependency (tests may run without the package installed)
try:
    from openai import OpenAI, AsyncOpenAI
    import openai as _openai_module
except ImportError:  # pragma: no cover - triggered in trimmed test envs
    OpenAI = None  # type: ignore
    AsyncOpenAI = None  # type: ignore
    _openai_module = None  # type: ignore
import httpx
import asyncio
import json

# Every prefix tag an LLM call can surface as response text instead of raising:
# all _classify_api_error() returns plus the "[API unavailable]" no-client
# sentinel. SINGLE source of truth — consumers (memory_storage, document
# generator, …) import this instead of hand-copying the list.
API_ERROR_PREFIXES: tuple = (
    "[API Error]",
    "[API unavailable]",
    "[CREDITS EXHAUSTED]",
    "[RATE LIMITED]",
    "[AUTH ERROR]",
    "[MODEL NOT SUPPORTED]",
    "[MODEL NOT FOUND]",
    "[SERVER ERROR]",
    # ResponseGenerator's streaming catch-all ("[Streaming Error: ...]" and
    # "[Streaming Error] ..."). Not produced by _classify_api_error but it is
    # response-text-instead-of-raise all the same; before 2026-08-14 it was
    # missing here and ~20 upstream-disconnect turns persisted as real replies.
    "[Streaming Error",
    # ResponseGenerator's reasoning-only recovery failed to produce visible
    # content twice. This is a model/transport failure, not an assistant reply.
    "[Error: Model returned empty response",
)


def _classify_api_error(e: Exception) -> str:
    """Classify an OpenAI/API error into a user-friendly message with a prefix tag.

    Returns a string like '[CREDITS EXHAUSTED] ...' or '[API Error] ...'
    that the GUI can pattern-match on for specific display. Every prefix
    returned here must appear in API_ERROR_PREFIXES above.
    """
    err_str = str(e).lower()
    err_code = getattr(e, 'code', '') or ''
    status_code = getattr(e, 'status_code', 0) or 0

    # --- Quota / billing exhaustion ---
    if (
        'insufficient_quota' in err_str
        or 'billing_hard_limit_reached' in err_str
        or 'exceeded your current quota' in err_str
        or err_code == 'insufficient_quota'
        or (status_code == 429 and 'quota' in err_str)
    ):
        return (
            "[CREDITS EXHAUSTED] You've run out of API credits. "
            "Please add credits at your provider's billing page, "
            "or switch to a different model."
        )

    # --- Provider credit exhaustion (OpenRouter HTTP 402) ---
    # "This request requires more credits, or fewer max_tokens. You requested
    # up to 9984 tokens, but can only afford 1985." Before 2026-08-21 this fell
    # through to the [API Error] fallback carrying the FULL raw error payload
    # (~3.2K of JSON incl. previous_errors), which was then streamed into the
    # chat bubble. Classify it and keep the message short and human.
    if (
        status_code == 402
        or 'requires more credits' in err_str
        or 'can only afford' in err_str
    ):
        return (
            "[CREDITS EXHAUSTED] The provider reports insufficient credits "
            "for this request (HTTP 402). Add credits at the provider's "
            "billing page, or switch to a cheaper model."
        )

    # --- Rate limiting (temporary) ---
    if status_code == 429 or 'rate_limit' in err_str or 'rate limit' in err_str:
        return (
            "[RATE LIMITED] API rate limit hit — too many requests. "
            "Wait a moment and try again, or switch to a different model."
        )

    # --- Authentication ---
    if (
        status_code == 401
        or 'invalid api key' in err_str
        or 'incorrect api key' in err_str
        or 'authentication' in err_str
    ):
        return (
            "[AUTH ERROR] API key is invalid or expired. "
            "Check your API key in .env or config.yaml."
        )

    # --- Model capability mismatch (e.g., image input not supported) ---
    if 'image input' in err_str or 'does not support' in err_str:
        return (
            f"[MODEL NOT SUPPORTED] The current model doesn't support this type of input "
            f"(e.g., images). Switch to a multimodal model like GPT-4o or Claude. ({e})"
        )

    # --- Model not found ---
    if status_code == 404 or 'model_not_found' in err_str or 'does not exist' in err_str:
        return f"[MODEL NOT FOUND] The requested model was not found. Check the model name in config. ({e})"

    # --- Server error ---
    if status_code >= 500:
        return f"[SERVER ERROR] The API provider is experiencing issues (HTTP {status_code}). Try again later."

    # --- Fallback ---
    return f"[API Error] {e}"


# Global embedding model cache to prevent re-loading SentenceTransformer
_global_embed_model = None
_embed_model_lock = asyncio.Lock()

# Global cross-encoder model cache to prevent re-loading CrossEncoder models
_global_cross_encoders = {}
_cross_encoder_lock = asyncio.Lock()


# Set OpenAI API key for API calls
# ---------------------------------------------------------------------------
# Model registry + capability matrix (SINGLE SOURCE OF TRUTH)
# ---------------------------------------------------------------------------
# Historically, "hooking up" a model meant adding it to api_models and then
# remembering to also touch four independent substring allowlists
# (supports_reasoning / supports_vision / supports_tools /
# supports_prompt_caching). Forgetting one silently disabled a feature for that
# model with NO error — e.g. a new model would parse fine but never call tools,
# or leak its chain-of-thought. That is the codebase's recurring "dead wiring"
# failure mode (see CLAUDE.md Critical Rule #3, tests/unit/test_tool_wiring_parity.py).
#
# Fix: declare every model's capabilities in ONE place (MODEL_CAPABILITIES),
# keyed by the full OpenRouter slug. The classifiers below derive their answer
# from pure slug-functions, and tests/unit/test_model_capability_wiring.py
# asserts (a) every registered slug has a capability row and (b) each classifier
# agrees with the declared row. Add a model → the parity test forces you to
# declare its caps, and any drift between a substring list and the declared
# intent fails loudly.
#
# Alias -> full OpenRouter slug. Module-level so the parity test can enumerate
# the roster without constructing a ModelManager (which loads local HF weights).
API_MODEL_ALIASES = {
    # Anthropic Claude (all are reasoning + vision + tools + explicit caching)
    # "claude-opus" is the generic alias; repointed 2026-07-22 from the retired
    # anthropic/claude-3-opus (no longer served on OpenRouter) to the current opus.
    "claude-opus": "anthropic/claude-opus-4.8",
    "claude-opus-4.5": "anthropic/claude-opus-4.5",
    "claude-opus-4.6": "anthropic/claude-opus-4.6",
    "claude-opus-4.7": "anthropic/claude-opus-4.7",
    "claude-opus-4.8": "anthropic/claude-opus-4.8",
    # Mythos-class (2026-06-09); ~2x Opus 4.8 price. Falls back to Opus 4.8 on
    # high-risk queries.
    "claude-fable-5": "anthropic/claude-fable-5",
    "fable-5": "anthropic/claude-fable-5",
    "sonnet-4.5": "anthropic/claude-sonnet-4.5",
    "sonnet-4.6": "anthropic/claude-sonnet-4.6",
    "haiku-4.5": "anthropic/claude-haiku-4.5",
    # OpenAI via OpenRouter
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4.1": "openai/gpt-4.1",
    "gpt-5": "openai/gpt-5",
    "gpt-5.1": "openai/gpt-5.1",
    "gpt-5.5": "openai/gpt-5.5",
    # GLM
    "glm-4.6": "z-ai/glm-4.6",
    "glm-4.7": "z-ai/glm-4.7",
    "glm-5": "z-ai/glm-5",
    "glm-5-turbo": "z-ai/glm-5-turbo",
    "glm-5.2": "z-ai/glm-5.2",
    # DeepSeek
    "deepseek-v3.1": "deepseek/deepseek-chat-v3.1",
    "deepseek-v4": "deepseek/deepseek-v4-pro",
    "deepseek-v4-flash": "deepseek/deepseek-v4-flash",
    "deepseek-r1": "deepseek/deepseek-r1-0528",
    # Moonshot AI (Kimi) via OpenRouter. Kimi K3: 2.8T-param multimodal reasoning
    # model, 1.05M ctx ($3/M in, $15/M out). "kimi-3" alias matches common usage.
    "kimi-k3": "moonshotai/kimi-k3",
    "kimi-3": "moonshotai/kimi-k3",
    "kimi-k2-thinking": "moonshotai/kimi-k2-thinking",
    "kimi-k2.6": "moonshotai/kimi-k2.6",
    # Google Gemini
    "gemini-3-pro": "google/gemini-3.1-pro-preview",
}

# Per-model capability truth, keyed by full slug.
#   reasoning: emits separable extended thinking (request reasoning separation)
#   vision:    accepts image input
#   tools:     supports function/tool calling (required by the agentic loop)
#   caching:   "explicit" = we inject cache_control breakpoints (Anthropic/GPT);
#              "implicit" = provider auto-caches server-side, no markers (Kimi);
#              None       = no prompt caching
_CLAUDE = {"reasoning": True, "vision": True, "tools": True, "caching": "explicit"}
_GPT = {"reasoning": False, "vision": True, "tools": True, "caching": "explicit"}
MODEL_CAPABILITIES = {
    "anthropic/claude-opus-4.5": _CLAUDE,
    "anthropic/claude-opus-4.6": _CLAUDE,
    "anthropic/claude-opus-4.7": _CLAUDE,
    "anthropic/claude-opus-4.8": _CLAUDE,
    "anthropic/claude-fable-5": _CLAUDE,
    "anthropic/claude-sonnet-4.5": _CLAUDE,
    "anthropic/claude-sonnet-4.6": _CLAUDE,
    "anthropic/claude-haiku-4.5": _CLAUDE,
    "openai/gpt-4o-mini": _GPT,
    "openai/gpt-4o": _GPT,
    "openai/gpt-4.1": _GPT,
    "openai/gpt-5": _GPT,
    "openai/gpt-5.1": _GPT,
    "openai/gpt-5.5": _GPT,
    # All GLM models on OpenRouter support tools (verified 2026-07-22).
    "z-ai/glm-4.6": {"reasoning": False, "vision": False, "tools": True, "caching": None},
    "z-ai/glm-4.7": {"reasoning": False, "vision": False, "tools": True, "caching": None},
    "z-ai/glm-5": {"reasoning": False, "vision": False, "tools": True, "caching": None},
    "z-ai/glm-5-turbo": {"reasoning": False, "vision": False, "tools": True, "caching": None},
    "z-ai/glm-5.2": {"reasoning": False, "vision": False, "tools": True, "caching": None},
    "deepseek/deepseek-chat-v3.1": {"reasoning": False, "vision": False, "tools": True, "caching": None},
    "deepseek/deepseek-v4-pro": {"reasoning": True, "vision": False, "tools": True, "caching": None},
    "deepseek/deepseek-v4-flash": {"reasoning": True, "vision": False, "tools": True, "caching": None},
    # deepseek-r1 tools=True: verified against OpenRouter's supported_parameters
    # (tools + tool_choice present) on 2026-07-22.
    "deepseek/deepseek-r1-0528": {"reasoning": True, "vision": False, "tools": True, "caching": None},
    # Kimi caches server-side automatically (implicit) — no cache_control markers.
    # forced_top_p: the K3 endpoint 400s on any top_p other than 0.95
    # ("Invalid value for 'top_p': 0.9. This endpoint requires top_p=0.95",
    # observed 2026-07-30) — resolve_top_p() overrides whatever the caller asked for.
    "moonshotai/kimi-k3": {"reasoning": True, "vision": True, "tools": True, "caching": "implicit", "forced_top_p": 0.95},
    "moonshotai/kimi-k2-thinking": {"reasoning": True, "vision": False, "tools": True, "caching": "implicit"},
    "moonshotai/kimi-k2.6": {"reasoning": True, "vision": True, "tools": True, "caching": "implicit"},
    # Gemini caching disabled pending OpenRouter support (see supports below).
    "google/gemini-3.1-pro-preview": {"reasoning": False, "vision": True, "tools": True, "caching": None},
}

# Context-window sizes by full slug — ONLY models whose limits are confidently
# known are listed; everything else falls back to DEFAULT_API_CONTEXT_LIMIT.
# (get_context_limit() previously hardcoded 128000 for ALL API models — the
# "Default GPT-4 Turbo context" comment survived three model generations and
# fed the token-budget computation a wrong ctx for every non-GPT model.)
# When adding entries, verify against OpenRouter /models context_length
# (extend scripts/verify_model_capabilities_live.py).
DEFAULT_API_CONTEXT_LIMIT = 128_000
MODEL_CONTEXT_LIMITS = {
    # Anthropic Claude: 200K standard tier across current models.
    **{slug: 200_000 for slug in MODEL_CAPABILITIES if slug.startswith("anthropic/claude")},
    "openai/gpt-4o": 128_000,
    "openai/gpt-4o-mini": 128_000,
    # Kimi K3: 1.05M ctx (see alias comment above).
    "moonshotai/kimi-k3": 1_050_000,
}


def _slug_forced_top_p(full_slug: str):
    """Provider-mandated top_p for a full slug, or None when any value is accepted.

    Some endpoints reject requests whose top_p differs from a fixed value instead
    of clamping it. Declared per model in MODEL_CAPABILITIES ("forced_top_p").
    """
    return MODEL_CAPABILITIES.get(str(full_slug), {}).get("forced_top_p")


def _slug_supports_reasoning(full_slug: str) -> bool:
    """Whether a full model slug emits separable extended thinking/reasoning."""
    s = str(full_slug).lower()
    if s.startswith("anthropic/claude"):
        return True
    if "deepseek-r1" in s or "deepseek-v4" in s:
        return True
    # Moonshot reasoning variants: K3 reasoning model, K2 Thinking, and K2.6
    # (OpenRouter reports `reasoning` in its supported_parameters).
    if "kimi-k3" in s or "kimi-k2-thinking" in s or "kimi-k2.6" in s:
        return True
    return False


def _slug_supports_vision(full_slug: str) -> bool:
    """Whether a full model slug accepts image/vision input."""
    s = str(full_slug).lower()
    # Text-only families never take image input.
    if "deepseek" in s or "glm" in s:
        return False
    # All Anthropic Claude models are multimodal — a startswith check (not a
    # per-name substring list) so new Claude models like fable-5 aren't missed.
    if s.startswith("anthropic/claude"):
        return True
    vision_patterns = ("gpt-4o", "gpt-4.1", "gpt-5", "gemini",
                       "kimi-k3", "kimi-k2.6", "kimi-k2.5")
    return any(p in s for p in vision_patterns)


def _slug_supports_tools(full_slug: str) -> bool:
    """Whether a full model slug supports function/tool calling."""
    s = str(full_slug).lower()
    # All Anthropic Claude models support tools (startswith, not per-name).
    if s.startswith("anthropic/claude"):
        return True
    tool_patterns = ("gpt-4", "gpt-5",
                     "deepseek-chat", "deepseek-coder", "deepseek-v4", "deepseek-r1",
                     "gemini", "glm", "kimi")
    return any(p in s for p in tool_patterns)


def _slug_supports_prompt_caching(full_slug: str) -> bool:
    """Whether we should inject explicit cache_control breakpoints for a slug.

    Implicit/server-side auto-caching models (DeepSeek, Kimi) return False —
    they cache automatically without markers, so we must NOT send cache_control.
    """
    s = str(full_slug).lower()
    if s.startswith("anthropic/claude"):
        return True
    if s.startswith("openai/gpt"):
        # gpt-4o, gpt-4o-mini, gpt-5+ and gpt-4.1+ support caching.
        if "gpt-4o" in s or "gpt-5" in s:
            return True
        if "gpt-4." in s:
            try:
                version = s.split("gpt-4.")[1].split("-")[0].split("/")[0]
                if float(version) >= 1:
                    return True
            except (IndexError, ValueError):
                pass
    # NOTE: Gemini caching temporarily disabled pending OpenRouter support.
    return False


class ModelManager:
    """Manager class for handling both local and API-based language models."""

    def __init__(self, api_key: str = None):
        # Active model name (local or API)
        self.active_model_name = None
        # Dictionary of loaded local models
        self.models = {}
        self.allow_fallback = False  # Disable fallback to unknown API models
        # Dictionary of loaded tokenizers for local models
        self.tokenizers = {}
        # Dictionary mapping registered API models
        self.api_models = {}
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

        self.base_url = "https://openrouter.ai/api/v1"

        # Initialize OpenAI clients only when the package is available AND an API key is provided.
        if OpenAI is not None and AsyncOpenAI is not None and self.api_key:
            sync_http_client = httpx.Client(
                timeout=httpx.Timeout(120.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=10),
                headers={"Connection": "keep-alive"},
            )

            async_http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=10),
                headers={"Connection": "keep-alive"},
            )

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=sync_http_client,
            )

            self.async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=async_http_client,
            )
        else:
            # No API key or OpenAI package unavailable — operate in stub/offline mode.
            self.client = None  # type: ignore[assignment]
            self.async_client = None  # type: ignore[assignment]

        self.default_model = "gpt-4-turbo"
        # Embedding model used across memory/gating paths. Use global cache to prevent re-loading
        self.embed_model = self._get_cached_embedder()
        # Runtime-overridable defaults (mutable via GUI and persisted to config)
        self.default_temperature = DEFAULT_TEMPERATURE
        self.default_max_tokens = DEFAULT_MAX_TOKENS
        # Register API model aliases from the module-level single source of truth.
        # Capabilities are declared alongside in MODEL_CAPABILITIES and enforced
        # by tests/unit/test_model_capability_wiring.py.
        self.api_models.update(API_MODEL_ALIASES)

    def reinitialize_clients(self, api_key: str = None) -> bool:
        """
        Reinitialize the OpenAI clients with a new API key.

        This is used by the wizard after the user enters their API key,
        since the ModelManager was created before the key was available.

        Args:
            api_key: The new API key. If None, reads from OPENROUTER_API_KEY or OPENAI_API_KEY env vars

        Returns:
            bool: True if clients were successfully initialized, False otherwise
        """
        new_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not new_key:
            logger.warning("[ModelManager] Cannot reinitialize clients: no API key provided")
            return False

        if OpenAI is None or AsyncOpenAI is None:
            logger.warning("[ModelManager] Cannot reinitialize clients: OpenAI package not available")
            return False

        try:
            # Close existing clients if any
            if self.client is not None:
                try:
                    self.client.close()
                except Exception:
                    pass

            # Create new clients with the new API key
            self.api_key = new_key

            sync_http_client = httpx.Client(
                timeout=httpx.Timeout(120.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=10),
                headers={"Connection": "keep-alive"},
            )

            async_http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=10),
                headers={"Connection": "keep-alive"},
            )

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=sync_http_client,
            )

            self.async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=async_http_client,
            )

            logger.info("[ModelManager] Clients reinitialized with new API key")
            return True

        except Exception as e:
            logger.error(f"[ModelManager] Failed to reinitialize clients: {e}")
            return False

    def _stub_response(self, prompt: str) -> str:
        return "[API unavailable] Unable to reach the language model. Please check your API key and network connection."

    def _stub_stream(self, prompt: str):
        async def _gen():
            yield self._stub_response(prompt)

        return _gen()
    def list_provider_models(self, vendor_prefix=None):
        """
        Returns a list of model ids exposed by the provider (OpenRouter).
        Optionally filter by a vendor prefix, e.g. 'anthropic/claude-3-opus'.
        """
        if self.client is None:
            return []
        r = self.client._client.get(f"{self.base_url}/models", headers={
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        })
        r.raise_for_status()
        data = r.json()
        models = [m["id"] for m in data.get("data", []) if "id" in m]
        if vendor_prefix:
            models = [m for m in models if m.startswith(vendor_prefix)]
        return models

    def get_context_limit(self):
        """Get the maximum context window based on active model.

        API models resolve through MODEL_CONTEXT_LIMITS (registry, keyed by
        full slug); unknown slugs fall back to DEFAULT_API_CONTEXT_LIMIT.
        """
        active = self.get_active_model_name()
        if self.is_api_model(active):
            # active may be an alias (key in api_models) OR an already-resolved
            # full slug — same dual-form caveat as supports_tools().
            full_model = (
                self.api_models[active] if active in self.api_models else str(active)
            )
            return MODEL_CONTEXT_LIMITS.get(full_model, DEFAULT_API_CONTEXT_LIMIT)
        model = self.get_model()
        if model:
            return model.config.max_position_embeddings
        else:
            raise ValueError("[ERROR] No model loaded. Cannot determine context limit.")

    @log_and_time("Get Embedder")
    def get_embedder(self):
        return self.embed_model

    @staticmethod
    def _get_cached_embedder():
        """Get or create a cached SentenceTransformer model to avoid re-loading"""
        global _global_embed_model

        if _global_embed_model is not None:
            return _global_embed_model

        try:
            logger.debug("Loading SentenceTransformer model (first time only)...")
            _global_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.debug("SentenceTransformer model cached successfully")
            return _global_embed_model
        except Exception:
            logger.warning("Failed to load SentenceTransformer, using stub embedder")
            class _StubEmbedder:
                def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
                    import numpy as np
                    n = len(texts or [])
                    return np.zeros((n, 384), dtype=np.float32)

            # Cache the stub embedder too to avoid re-creating it
            _global_embed_model = _StubEmbedder()
            return _global_embed_model

    @log_and_time("Get Cross-Encoder")
    def get_cross_encoder(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Get or create a cached CrossEncoder model to avoid re-loading.

        Args:
            model_name: Name of the cross-encoder model to load

        Returns:
            CrossEncoder: Cached or newly created cross-encoder instance
        """
        global _global_cross_encoders

        # Check cache first
        if model_name in _global_cross_encoders:
            logger.debug(f"Using cached cross-encoder: {model_name}")
            return _global_cross_encoders[model_name]

        try:
            logger.debug(f"Loading CrossEncoder model (first time only): {model_name}")
            from sentence_transformers import CrossEncoder

            cross_encoder = CrossEncoder(model_name)

            # Cache the model for future use
            _global_cross_encoders[model_name] = cross_encoder
            logger.debug(f"CrossEncoder model cached successfully: {model_name}")

            return cross_encoder

        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder '{model_name}': {e}")
            # Return a stub cross-encoder that always returns neutral scores
            class _StubCrossEncoder:
                def predict(self, pairs, batch_size=32, show_progress_bar=False):
                    import numpy as np
                    n = len(pairs or [])
                    return np.ones(n, dtype=np.float32) * 0.5  # Neutral score

            # Cache the stub cross-encoder too to avoid re-creating it
            stub_encoder = _StubCrossEncoder()
            _global_cross_encoders[model_name] = stub_encoder
            return stub_encoder

    @log_and_time("Load Model")
    def load_model(self, model_name, model_path):
        """Load a local Huggingface model and tokenizer."""
        try:
            # Determine if using local files only
            local_files_only = model_path.startswith("./")

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=local_files_only,
                device_map="auto",
                trust_remote_code=True
            )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=local_files_only,
                trust_remote_code=True,
                use_fast=True
            )

            # Ensure tokenizer has a pad token set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Set tokenizer max length to model context window
            tokenizer.model_max_length = model.config.max_position_embeddings

            # Store model and tokenizer
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer

            # logger.debug( Successfully loaded model: {model.__class__.__name__}")
        except Exception as e:
            raise ValueError(f"Error loading model '{model_name}': {str(e)}")
    @log_and_time("Load OpenAI")
    def load_openai_model(self, model_name, api_model_name):
        """Register an OpenAI API model (no local loading)."""
        # logger.debug( Registering OpenAI model '{model_name}'")
        self.api_models[model_name] = api_model_name
    def close(self):
        """Gracefully close the HTTP client to avoid socket leak."""
        if hasattr(self.client, "_client"):
            self.client._client.close()
    async def aclose(self):
        if hasattr(self.async_client, "_client"):
            await self.async_client._client.aclose()
    def is_api_model(self, model_name):
        """Check if a given model is an API-based model."""
        return model_name in self.api_models

    def supports_reasoning(self, model_name: str) -> bool:
        """Check if a model may return extended thinking / reasoning content.

        When True, API calls should request reasoning separation via extra_body
        so that thinking content doesn't leak into the visible response.
        """
        if model_name not in self.api_models:
            return False
        return _slug_supports_reasoning(self.api_models[model_name])

    def resolve_top_p(self, model_name, requested=None):
        """Effective top_p for a request against `model_name` (alias or full slug).

        A model with a declared forced_top_p (MODEL_CAPABILITIES) always gets that
        value — its endpoint rejects anything else — regardless of what the caller
        asked for. Otherwise the caller's value, falling back to DEFAULT_TOP_P.
        """
        slug = self.api_models.get(model_name, model_name)
        forced = _slug_forced_top_p(slug)
        if forced is not None:
            if requested is not None and requested != forced:
                logger.debug(
                    f"[ModelManager] {slug} requires top_p={forced}; "
                    f"overriding requested {requested}"
                )
            return forced
        return DEFAULT_TOP_P if requested is None else requested

    def supports_prompt_caching(self, model_name):
        """Check if a given API model supports EXPLICIT prompt caching.

        True only for models that need cache_control breakpoints injected
        (Anthropic, recent GPT). Implicit/server-side auto-caching models
        (DeepSeek, Kimi) return False — they cache without markers.
        """
        if model_name not in self.api_models:
            return False
        return _slug_supports_prompt_caching(self.api_models[model_name])

    @staticmethod
    def _strip_cache_breakpoint(text):
        """Remove the prompt-cache breakpoint marker from a system prompt.

        Used on every path that does NOT split the prompt into cached/uncached
        blocks (non-cacheable models, image requests, tool calls, fallbacks), so
        the marker never reaches the model. No-op when the marker is absent.
        """
        if text and PROMPT_CACHE_BREAKPOINT in text:
            return text.replace(PROMPT_CACHE_BREAKPOINT, "")
        return text

    def _format_messages_with_cache(self, system_prompt, user_prompt):
        """
        Format messages with cache_control breakpoints for supported models.

        The system prompt is split at PROMPT_CACHE_BREAKPOINT: the stable prefix
        before the marker carries cache_control (cached across turns), and the
        per-turn volatile tail after the marker is a second system block with NO
        cache_control (it changes every turn and would otherwise invalidate the
        whole cached prefix). The user prompt is never cached. When the marker is
        absent (callers without per-turn appends), the entire system prompt is
        cached as a single block — backward compatible.

        Args:
            system_prompt: The system prompt (optionally containing the marker)
            user_prompt: The user prompt (not cached)

        Returns:
            List of message dictionaries with cache_control breakpoints
        """
        stable, sep, volatile = (system_prompt or "").partition(PROMPT_CACHE_BREAKPOINT)
        system_content = [
            {
                "type": "text",
                "text": stable,
                "cache_control": {"type": "ephemeral"}
            }
        ]
        # Per-turn content after the breakpoint — appended uncached so it can
        # change every turn without busting the cached prefix above.
        if sep and volatile.strip():
            system_content.append({
                "type": "text",
                "text": volatile
            })
        return [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

    def _log_cache_usage(self, usage, *, model_name=None, where=""):
        """Log prompt-cache + token usage from an API response's ``usage`` object.

        Best-effort and fully exception-safe — never raises into the generation
        path. We talk to Anthropic models through OpenRouter's OpenAI-compatible
        API, so cache stats can surface under several shapes; this reads all of
        them: the OpenAI-normalised ``prompt_tokens_details.cached_tokens``, the
        Anthropic-native passthrough fields (``cache_read_input_tokens`` /
        ``cache_creation_input_tokens``), and OpenRouter's ``cache_discount``.

        Emits a single greppable ``[PromptCache]`` INFO line per call:
          - ``HIT``   — tokens were served from cache (cheap read)
          - ``WRITE`` — tokens were written to cache this turn (~1.25x premium)
          - ``MISS``  — no cache activity (neither read nor write)
        """
        if usage is None:
            return
        try:
            if hasattr(usage, "model_dump"):
                d = usage.model_dump()
            elif isinstance(usage, dict):
                d = usage
            else:
                d = {
                    k: getattr(usage, k)
                    for k in dir(usage)
                    if not k.startswith("_") and not callable(getattr(usage, k, None))
                }
        except Exception:
            return

        try:
            prompt = d.get("prompt_tokens")
            completion = d.get("completion_tokens")
            ptd = d.get("prompt_tokens_details") or {}
            if not isinstance(ptd, dict):
                ptd = getattr(ptd, "__dict__", {}) or {}
            cached_read = ptd.get("cached_tokens")
            # Anthropic-native passthrough (present on some OpenRouter responses)
            cache_read = d.get("cache_read_input_tokens")
            cache_creation = d.get("cache_creation_input_tokens")
            cache_discount = d.get("cache_discount")

            served = next((v for v in (cache_read, cached_read) if v), 0) or 0
            written = cache_creation or 0
            status = "HIT" if served else ("WRITE" if written else "MISS")

            logger.info(
                "[PromptCache] %s model=%s where=%s prompt=%s completion=%s "
                "cache_read=%s cache_write=%s cache_discount=%s",
                status, model_name or "?", where or "?", prompt, completion,
                served, written, cache_discount,
            )
        except Exception as e:  # pragma: no cover - diagnostic only
            logger.debug("[PromptCache] usage parse failed: %s", e)

    @log_and_time("Generate with openAI")
    def generate_with_openai(self, prompt, model_name, system_prompt=None, max_tokens=None, temperature=None, top_p=None):
        """Generate text using OpenAI API, with global defaults fallback."""
        # Apply global defaults if not provided (allow runtime override)
        max_tokens = (self.default_max_tokens if max_tokens is None else max_tokens)
        temperature = (self.default_temperature if temperature is None else temperature)
        top_p = self.resolve_top_p(model_name, top_p)

        if self.client is None:
            return self._stub_response(prompt)

        try:
            # Stop sequences to prevent hallucinating user responses (less restrictive)
            stop_sequences = [
                "\n\nUser:",
                "\n\nUSER:",
                "\n\nHuman:",
            ]

            # Check if model supports caching and format messages accordingly
            # Need to check against the alias name (e.g., "claude-opus" not "anthropic/claude-3-opus")
            model_alias = None
            for alias, full_name in self.api_models.items():
                if full_name == model_name:
                    model_alias = alias
                    break

            if model_alias and self.supports_prompt_caching(model_alias):
                logger.debug(f"Using prompt caching for model: {model_name}")
                messages = self._format_messages_with_cache(
                    system_prompt or SYSTEM_PROMPT,
                    prompt
                )
            else:
                messages = [
                    {"role": "system", "content": self._strip_cache_breakpoint(system_prompt or SYSTEM_PROMPT)},
                    {"role": "user", "content": prompt}
                ]

            # logger.debug(  Calling OpenAI API: {model_name}")
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_sequences,
                # Ask OpenRouter to include usage accounting (cost + cache stats)
                # in the response, so prompt-cache hits are observable.
                extra_body={"usage": {"include": True}},
            )

            self._log_cache_usage(
                getattr(response, "usage", None),
                model_name=model_name,
                where="generate_with_openai",
            )

            content = response.choices[0].message.content
            return content.strip() if content else ""

        except Exception as e:
            return _classify_api_error(e)

    def switch_model(self, model_name):
        """Switch active model (local or API)."""
        self.active_model_name = model_name

    def get_model(self):
        """Return active local model instance (if any)."""
        return self.models.get(self.active_model_name)

    def get_tokenizer(self):
        """Return active tokenizer instance (if any)."""
        return self.tokenizers.get(self.active_model_name)

    def get_active_model_name(self):
        """Return the name of the currently active model."""
        return self.active_model_name

    @staticmethod
    def truncate_prompt(prompt, tokenizer, max_input_tokens, preserve_prefix="You are Daemon"):
        """Ensure prompt fits within model's input size (optional prefix preservation)."""
        if preserve_prefix in prompt:
            prefix_index = prompt.index(preserve_prefix)
            prefix = prompt[:prefix_index + len(preserve_prefix)]
            rest = prompt[prefix_index + len(preserve_prefix):]
        else:
            prefix = ""
            rest = prompt

        prefix_tokens = tokenizer.encode(prefix)
        rest_tokens = tokenizer.encode(rest)

        # No truncation needed
        if len(prefix_tokens) + len(rest_tokens) <= max_input_tokens:
            return prompt

        # logger.debug( Truncating prompt: {len(prefix_tokens) + len(rest_tokens)} → {max_input_tokens} tokens")

        # If prefix alone is too long, truncate from the end of entire prompt
        allowed_rest_tokens = max_input_tokens - len(prefix_tokens)
        if allowed_rest_tokens <= 0:
            print("[WARN] Prefix alone exceeds max input size.")
            return tokenizer.decode((prefix_tokens + rest_tokens)[-max_input_tokens:])

        # Truncate rest of prompt and return combined prompt
        truncated_rest_tokens = rest_tokens[-allowed_rest_tokens:]
        return tokenizer.decode(prefix_tokens + truncated_rest_tokens)



    @log_and_time("ModelManager Generate Call")
    def generate(self, prompt, model_name="gpt-4-turbo", max_tokens=None, temperature=None, top_p=None, top_k=None, no_repeat_ngram_size=None, pad_token_id=None, system_prompt=None):
        """Main generate function for both local and OpenAI models."""

        # IMPORTANT: Use the provided model_name if given, otherwise use active model
        logger.debug(f"[generate] Model received: {model_name}")
        logger.debug(f"[generate] Known local models: {list(self.models.keys())}")
        logger.debug(f"[generate] Known API models: {list(self.api_models.keys())}")

        target_model = model_name or self.active_model_name

        if not target_model:
            raise ValueError("No model specified. Pass model_name or use switch_model() first.")

        # Check if this is a local model FIRST
        if target_model in self.models:  # This is a local model
            logger.debug(f"Using local model: {target_model}")

            # Get the specific local model and tokenizer
            model = self.models[target_model]
            tokenizer = self.tokenizers[target_model]

            # Use defaults where needed
            max_tokens = 64 if max_tokens is None else max_tokens
            temperature = self.default_temperature if temperature is None else temperature
            top_p = DEFAULT_TOP_P if top_p is None else top_p
            top_k = DEFAULT_TOP_K if top_k is None else top_k

            # Check input length and truncate if needed
            tokens = tokenizer.encode(prompt)
            max_len = model.config.max_position_embeddings

            if len(tokens) > max_len:
                tokens = tokens[-max_len:]
                prompt = tokenizer.decode(tokens)

            # Prepare safe prompt for max input size
            context_limit = model.config.max_position_embeddings
            max_input_tokens = context_limit - max_tokens
            safe_prompt = self.truncate_prompt(prompt, tokenizer, max_input_tokens)

            # Tokenize and move inputs to model device
            inputs = tokenizer(safe_prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate output with Huggingface model
            with torch.no_grad():
                start = time.time()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    # Enable sampling so temperature has an effect on local models
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=1,
                    pad_token_id=pad_token_id or tokenizer.pad_token_id
            )
                end = time.time()
                logger.debug(f"Local generation finished in {end - start:.2f} seconds")

            # Decode and return output text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from the output
            return generated_text[len(safe_prompt):].strip()

        # API model generation path
        elif target_model in self.api_models:
            logger.debug(f"Using OpenAI model alias: {self.api_models[target_model]}")
            return self.generate_with_openai(
                prompt,
                self.api_models[target_model],
                system_prompt=system_prompt,
                max_tokens=max_tokens or 500,
                temperature=self.default_temperature if temperature is None else temperature,
                top_p=top_p or 1.0
            )

        else:
            if not self.allow_fallback:
                raise ValueError(f"[ModelManager] Model '{target_model}' is not recognized as a local or registered API model. Fallback is disabled.")

            logger.warning(f"[ModelManager] Fallback triggered for unknown model: {target_model}")
            return self.generate_with_openai(
                prompt,
                target_model,
                system_prompt=system_prompt,
                max_tokens=max_tokens or 500,
                temperature=temperature or 0.7,
                top_p=top_p or 1.0
            )
    async def generate_once(self,
                            prompt: str,
                            model_name: str = None,
                            system_prompt: str = "You are a concise and helpful assistant.",
                            max_tokens: int = 256,
                            temperature: float = None,
                            top_p: float = None,
                            disable_reasoning: bool = False) -> str:
        """
        Generates a single, complete response asynchronously (non-streaming).
        Ideal for internal tasks like query rewriting or classification.

        disable_reasoning: when True, suppress native reasoning separation even for
        reasoning-capable models. Used as a recovery retry when a reasoning model
        (e.g. deepseek-v4) swallows the whole answer into the reasoning channel and
        returns empty visible content — disabling reasoning forces it to emit the
        answer as normal content.
        """
        target_model = model_name or self.active_model_name
        if not target_model:
            logger.error("[generate_once] No model specified or active.")
            raise ValueError("No model specified. Pass model_name or use switch_model() first.")

        # --- Handle Local Models ---
        if target_model in self.models:
            # Use the existing synchronous generate method in a separate thread
            return await asyncio.to_thread(
                self.generate,
                prompt,
                model_name=target_model,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature if temperature is not None else self.default_temperature,
                top_p=top_p if top_p is not None else DEFAULT_TOP_P
            )

        # --- Handle API Models ---
        elif target_model in self.api_models:
            if self.async_client is None:
                return self._stub_response(prompt)
            try:
                # Stop sequences to prevent hallucinating user responses (less restrictive)
                stop_sequences = [
                    "\n\nUser:",
                    "\n\nUSER:",
                    "\n\nHuman:",
                ]

                # Check if model supports caching and format messages accordingly
                if self.supports_prompt_caching(target_model):
                    logger.debug(f"Using prompt caching for model: {target_model}")
                    messages = self._format_messages_with_cache(system_prompt, prompt)
                else:
                    messages = [
                        {"role": "system", "content": self._strip_cache_breakpoint(system_prompt)},
                        {"role": "user", "content": prompt}
                    ]

                create_kwargs = dict(
                    model=self.api_models[target_model],
                    messages=messages,
                    max_tokens=(max_tokens if max_tokens is not None else self.default_max_tokens),
                    temperature=temperature if temperature is not None else self.default_temperature,
                    top_p=self.resolve_top_p(target_model, top_p),
                    stop=stop_sequences,
                    stream=False,
                )

                # Ask OpenRouter for usage accounting so prompt-cache stats
                # (cached_tokens, cache_discount) appear in the usage object —
                # without this, caching is invisible even when it happens.
                create_kwargs["extra_body"] = {"usage": {"include": True}}

                # Request native reasoning separation for supported models
                # (unless the caller explicitly disabled it for a recovery retry)
                if self.supports_reasoning(target_model):
                    # Omitting the reasoning param does NOT disable reasoning
                    # for reasoning-by-default models: kimi-k3 kept reasoning
                    # through the 2026-08-31 insight assessor call with no
                    # reasoning key sent and blew the 75s timeout. OpenRouter's
                    # explicit off-switch is enabled=false.
                    create_kwargs["extra_body"]["reasoning"] = (
                        {"enabled": False} if disable_reasoning
                        else {"effort": "medium"}
                    )

                response = await self.async_client.chat.completions.create(**create_kwargs)

                self._log_cache_usage(
                    getattr(response, "usage", None),
                    model_name=self.api_models[target_model],
                    where="generate_once",
                )

                msg = response.choices[0].message
                content = msg.content

                # Handle content returned as list of content blocks
                # (Anthropic extended thinking via some providers)
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                        elif hasattr(block, "type") and block.type == "text":
                            text_parts.append(getattr(block, "text", ""))
                    content = "\n".join(text_parts) if text_parts else str(content)

                return content.strip() if content else ""

            except Exception as e:
                classified = _classify_api_error(e)
                logger.error(f"[ModelManager] OpenAI generate_once error: {classified}")
                return classified

        # --- Handle Unrecognized Models ---
        else:
            logger.error(f"[generate_once] Model '{target_model}' is not recognized.")
            raise ValueError(f"[ModelManager] Model '{target_model}' is not recognized as a local or registered API model.")

    def supports_vision(self, model_name: str = None) -> bool:
        """Check if a model supports image/vision input.

        Args:
            model_name: Model name to check. Uses active model if None.

        Returns:
            bool: True if model supports multimodal image input
        """
        target_model = model_name or self.active_model_name
        if not target_model:
            return False

        # Local models don't support vision
        if target_model in self.models:
            return False

        if target_model in self.api_models:
            return _slug_supports_vision(self.api_models[target_model])

        return False

    def supports_tools(self, model_name: str = None) -> bool:
        """
        Check if a model supports function/tool calling.

        Args:
            model_name: Model name to check. Uses active model if None.

        Returns:
            bool: True if model supports tool calling
        """
        target_model = model_name or self.active_model_name
        if not target_model:
            return False

        # Local models don't support tool calling
        if target_model in self.models:
            return False

        # Resolve to the full provider model name. target_model may be an ALIAS
        # (a key in api_models) OR an already-resolved name (e.g.
        # "deepseek/deepseek-v4-pro") — the agentic path passes the latter via
        # get_active_model_name(). Handle both, or tools get silently dropped and
        # the model can only narrate instead of calling propose_action/etc.
        full_model = (
            self.api_models[target_model]
            if target_model in self.api_models
            else str(target_model)
        )

        return _slug_supports_tools(full_model)

    async def generate_once_with_tools(
        self,
        prompt: str,
        model_name: str = None,
        system_prompt: str = "You are a helpful assistant.",
        tools: list = None,
        tool_choice: str = "auto",
        max_tokens: int = 500,
        temperature: float = 0.3,
    ):
        """
        Generate a response with function/tool calling support.

        Used for agentic workflows where the model can request actions
        (like web searches) via tool calls.

        Args:
            prompt: The user prompt
            model_name: Model to use (default: active model)
            system_prompt: System prompt
            tools: List of tool definitions (OpenAI format)
            tool_choice: "auto", "none", or specific tool name
            max_tokens: Maximum tokens for response
            temperature: Temperature for generation

        Returns:
            Raw response message object with potential tool_calls attribute.
            Returns a dict with 'content' key if tools not supported.
        """
        target_model = model_name or self.active_model_name
        if not target_model:
            logger.error("[generate_once_with_tools] No model specified or active.")
            raise ValueError("No model specified. Pass model_name or use switch_model() first.")

        # Check if model supports tools
        if not self.supports_tools(target_model):
            logger.warning(f"[generate_once_with_tools] Model {target_model} doesn't support tools, using standard generation")
            response_text = await self.generate_once(
                prompt=prompt,
                model_name=target_model,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return {"content": response_text, "tool_calls": None}

        # Handle API models with tool support
        if target_model in self.api_models:
            if self.async_client is None:
                return {"content": self._stub_response(prompt), "tool_calls": None}

            try:
                messages = [
                    {"role": "system", "content": self._strip_cache_breakpoint(system_prompt)},
                    {"role": "user", "content": prompt}
                ]

                # Build request parameters
                request_params = {
                    "model": self.api_models[target_model],
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                }

                # Add tools if provided
                if tools:
                    request_params["tools"] = tools
                    request_params["tool_choice"] = tool_choice

                # Request native reasoning separation for supported models — but NOT
                # when tools are in play. Reasoning models (DeepSeek) otherwise spend
                # the turn on chain-of-thought ("let me create it…") and never emit a
                # structured tool_call. Tool selection should be a direct, non-reasoning
                # decision (mirrors _generate_decision_no_reasoning on the XML path).
                # Ask OpenRouter for usage accounting (cost + cache stats).
                request_params["extra_body"] = {"usage": {"include": True}}

                if not tools and self.supports_reasoning(target_model):
                    request_params["extra_body"]["reasoning"] = {"effort": "medium"}

                response = await self.async_client.chat.completions.create(**request_params)

                # Return the message object (has content and tool_calls)
                return response.choices[0].message

            except Exception as e:
                classified = _classify_api_error(e)
                logger.error(f"[generate_once_with_tools] Error: {classified}")
                return {"content": classified, "tool_calls": None}

        # Unrecognized model
        logger.error(f"[generate_once_with_tools] Model '{target_model}' not recognized")
        raise ValueError(f"Model '{target_model}' is not recognized")

    def _format_user_content_with_images(self, text: str, images: list) -> list:
        """
        Format user content with images for multimodal API calls.

        Args:
            text: The text prompt
            images: List of image dicts with 'data' (base64), 'media_type', and 'filename'

        Returns:
            List of content blocks suitable for OpenAI/Anthropic multimodal APIs
        """
        content = [{"type": "text", "text": text}]

        for img in images:
            if not img.get("data") or not img.get("media_type"):
                continue

            # Format for OpenAI/Anthropic vision API
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img['media_type']};base64,{img['data']}",
                    "detail": "auto"  # Let API decide resolution
                }
            })

        logger.debug(f"[ModelManager] Formatted {len(images)} images for multimodal call")
        return content

    @log_and_time("ModelManager Generate Async")
    async def generate_async(self, prompt, raw=False, images=None, **kwargs):
        """
        Async wrapper for generation using the active model.

        Args:
            prompt: Text prompt
            raw: If True, skip system prompt formatting
            images: Optional list of image dicts for multimodal models
                    Each dict should have 'data' (base64), 'media_type', 'filename'
            **kwargs: Additional args (system_prompt, max_tokens, temperature, etc.)
        """
        # Recovery flag: suppress native reasoning separation even for reasoning
        # models. Popped so it never leaks into local-model / create() kwargs.
        disable_reasoning = bool(kwargs.pop('disable_reasoning', False))

        target_model = self.active_model_name  # No longer allows override
        logger.debug(f"[generate_async] Active model: {target_model}")
        logger.debug(f"[generate_async] Registered OpenAI models: {self.api_models}")
        logger.debug(f"[generate_async] Registered local models: {self.models}")

        # Drop images for models that don't support vision input
        if images and not self.supports_vision(target_model):
            logger.warning(f"[generate_async] Dropping {len(images)} images — {target_model} does not support vision input")
            images = None

        if images:
            total_size = sum(len(img.get("data", "")) for img in images)
            logger.warning(f"[generate_async] {len(images)} images included for multimodal processing, total base64={total_size//1024}KB")
        else:
            logger.warning(f"[generate_async] No images parameter received")

        if target_model in self.models:
            return await asyncio.to_thread(
                self.generate, prompt, model_name=target_model, **kwargs
            )
        elif target_model in self.api_models:
            if self.async_client is None:
                return self._stub_stream(prompt)

            try:
                logger.debug(f"[ModelManager] Using OpenAI async model: {target_model}")

                # Build messages based on raw mode and system prompt
                if raw:
                    # Handle images in raw mode
                    if images:
                        user_content = self._format_user_content_with_images(prompt, images)
                        messages = [{"role": "user", "content": user_content}]
                    else:
                        messages = [{"role": "user", "content": prompt}]
                else:
                    system_prompt_text = kwargs.get('system_prompt')
                    if system_prompt_text is None:
                        system_prompt_text = SYSTEM_PROMPT

                    # Format user content (with images if provided)
                    if images:
                        user_content = self._format_user_content_with_images(prompt, images)
                    else:
                        user_content = prompt

                    # Check if model supports caching and format messages accordingly
                    if self.supports_prompt_caching(target_model):
                        logger.debug(f"Using prompt caching for model: {target_model}")
                        # Note: caching with images may need special handling
                        if images:
                            # For now, skip caching when images are present
                            messages = [
                                {"role": "system", "content": self._strip_cache_breakpoint(system_prompt_text)},
                                {"role": "user", "content": user_content}
                            ]
                        else:
                            messages = self._format_messages_with_cache(system_prompt_text, prompt)
                    else:
                        messages = [
                            {"role": "system", "content": self._strip_cache_breakpoint(system_prompt_text)},
                            {"role": "user", "content": user_content}
                        ]

                for i, msg in enumerate(messages):
                    logger.debug(f"--- Prompt Message {i} ---")
                    logger.debug(f"Role: {msg['role']}")
                    content_preview = str(msg['content'])[:200] if isinstance(msg['content'], str) else "Complex content structure"
                    logger.debug(f"Content: {content_preview}...")

                # Stop sequences to prevent hallucinating user responses (less restrictive)
                stop_sequences = [
                    "\n\nUser:",
                    "\n\nUSER:",
                    "\n\nHuman:",
                ]

                create_kwargs = dict(
                    model=self.api_models[target_model],
                    messages=messages,
                    max_tokens=kwargs.get('max_tokens', self.default_max_tokens),
                    temperature=kwargs.get('temperature', self.default_temperature),
                    top_p=self.resolve_top_p(target_model, kwargs.get('top_p')),
                    stop=stop_sequences,
                    stream=True,
                    # Ask the provider to emit a trailing usage chunk so the
                    # consumer can log prompt-cache stats for the streaming path.
                    stream_options={"include_usage": True},
                )

                # Request native reasoning separation for models that support it
                # (Claude, DeepSeek-R1, Kimi K3). Thinking arrives via
                # delta.reasoning_content instead of being mixed into the text
                # response — so it's requested even on image turns (multimodal
                # reasoning models like Kimi K3 / Claude / Gemini support both
                # extended thinking and image input on the same endpoint).
                # Ask OpenRouter for usage accounting so prompt-cache stats are
                # observable on the streaming (main) path.
                create_kwargs["extra_body"] = {"usage": {"include": True}}

                if self.supports_reasoning(target_model) and not disable_reasoning:
                    create_kwargs["extra_body"]["reasoning"] = {"effort": "medium"}
                    logger.info(f"[generate_async] Enabled native reasoning for {target_model}"
                                + (" (images present)" if images else ""))
                elif disable_reasoning and self.supports_reasoning(target_model):
                    # Explicit off-switch — omission alone leaves
                    # reasoning-by-default models (kimi-k3) reasoning anyway.
                    create_kwargs["extra_body"]["reasoning"] = {"enabled": False}
                    logger.info(f"[generate_async] Native reasoning disabled by caller for {target_model} (recovery retry)")

                stream = await self.async_client.chat.completions.create(**create_kwargs)
                return stream
            except Exception as e:
                classified = _classify_api_error(e)
                logger.error(f"[ModelManager] OpenAI streaming error: {classified}")
                async def _error_stream():
                    yield classified
                return _error_stream()
        else:
            return await asyncio.to_thread(
                self.generate, prompt, model_name=target_model, **kwargs
            )
