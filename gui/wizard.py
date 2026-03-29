"""
gui/wizard.py

Conversational onboarding wizard for first-run setup.

Module Contract
- Purpose: Conversational wizard state machine for first-run onboarding. Collects API key,
  Tavily key, style preferences, and user identity through chat interface.
- Inputs:
  - WizardState dataclass tracking current step and collected data
  - process_wizard_message(user_input, state, orchestrator) → (response, new_state, is_complete)
  - get_welcome_message() → str
- Outputs:
  - Daemon's wizard responses (friendly, conversational)
  - Updated WizardState after each step
  - is_complete=True when wizard finishes
- Side effects:
  - Writes API keys to .env file (OPENAI_API_KEY, TAVILY_API_KEY)
  - Extracts user + entity facts via LLMFactExtractor (dual-budget, fact_scope aware)
  - Stores user facts to both UserProfile (JSON) AND ChromaDB (semantic retrieval)
  - Stores entity facts to ChromaDB only (not UserProfile)
  - Generates custom_personality.txt for warm/direct styles (balanced uses default)
  - Saves completed profile via UserProfile.save()
- Note: FactVerifier is intentionally skipped — no existing facts to conflict on first boot.
"""

import os
import re
import sys
from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

from utils.logging_utils import get_logger

logger = get_logger("wizard")


class WizardStep(Enum):
    """Wizard flow steps."""
    WELCOME = "welcome"
    INTRO = "intro"
    API_KEY = "api_key"
    TAVILY_KEY = "tavily_key"
    STYLE = "style"
    NAME = "name"
    PRONOUNS = "pronouns"
    BACKGROUND = "background"
    COMPLETE = "complete"


@dataclass
class WizardState:
    """Tracks wizard progress and collected data."""
    step: WizardStep = WizardStep.WELCOME
    collected_data: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    max_retries: int = 3


def get_welcome_message() -> str:
    """Get the initial wizard welcome message."""
    return """Hey! Looks like this is your first time here. I'm Daemon.

Before we get started, let me tell you a bit about what I am and how I work. Type anything to continue."""


def get_intro_message() -> str:
    """Get the introduction explaining what Daemon is and how it works."""
    return """**What is Daemon?**

I'm a conversational AI with persistent memory. Unlike typical chatbots that forget everything between sessions, I'm designed to remember our conversations, learn about you over time, and build a deeper understanding of who you are and what matters to you.

**How Memory Works**

Every conversation we have gets stored in a multi-tier memory system. I extract facts, track conversation threads, and periodically reflect on what I've learned. This means my responses improve over time — the more we talk, the better I understand your context, preferences, and history.

**Privacy & Data**

Right now, your conversations pass through whichever LLM provider you choose (OpenRouter gives you access to Claude, GPT-4, and many others). However, Daemon is architected for full local operation — with a 4090 or better GPU, you can swap in local models and keep 100% of your data on your own machine. The memory system itself already runs entirely locally.

**What makes Daemon different**

- **I don't forget.** Facts about you, your preferences, and our history persist across sessions.
- **I reflect.** Periodically I consolidate what I've learned into summaries and insights.
- **I improve.** A denser memory space and ongoing reflection mean I get better at helping you over time.

Type anything to continue to setup."""


def write_api_key_to_env(key: str) -> bool:
    """
    Write API key to .env file as plaintext.
    Standard practice - security via filesystem permissions and .gitignore.
    Args:
        key: API key to write
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if getattr(sys, 'frozen', False):
            env_path = Path(os.environ.get('APPDATA', '')) / 'Daemon' / '.env'
        else:
            env_path = Path('.env')
        lines = []

        # Read existing .env if it exists
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

        # Update or append OPENAI_API_KEY line
        key_found = False
        for i, line in enumerate(lines):
            if line.startswith('OPENAI_API_KEY='):
                lines[i] = f'OPENAI_API_KEY={key}\n'
                key_found = True
                break

        if not key_found:
            # Ensure newline before appending
            if lines and not lines[-1].endswith('\n'):
                lines.append('\n')
            lines.append(f'OPENAI_API_KEY={key}\n')

        # Write atomically
        with open(env_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        # Also set in current environment for immediate effect
        os.environ['OPENAI_API_KEY'] = key

        logger.info("[Wizard] API key written to .env and set in environment")
        return True

    except Exception as e:
        logger.error(f"[Wizard] Failed to write API key to .env: {e}")
        return False


def validate_api_key_format(key: str) -> bool:
    """
    Check if key matches OpenRouter format.

    Args:
        key: API key to validate

    Returns:
        bool: True if format is valid
    """
    key = key.strip()
    return key.startswith('sk-or-') and len(key) > 20


def write_tavily_key_to_env(key: str) -> bool:
    """
    Write Tavily API key to .env file as plaintext.

    Args:
        key: Tavily API key to write
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if getattr(sys, 'frozen', False):
            env_path = Path(os.environ.get('APPDATA', '')) / 'Daemon' / '.env'
        else:
            env_path = Path('.env')
        lines = []

        # Read existing .env if it exists
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

        # Update or append TAVILY_API_KEY line
        key_found = False
        for i, line in enumerate(lines):
            if line.startswith('TAVILY_API_KEY='):
                lines[i] = f'TAVILY_API_KEY={key}\n'
                key_found = True
                break

        if not key_found:
            # Ensure newline before appending
            if lines and not lines[-1].endswith('\n'):
                lines.append('\n')
            lines.append(f'TAVILY_API_KEY={key}\n')

        # Write atomically
        with open(env_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        # Also set in current environment for immediate effect
        os.environ['TAVILY_API_KEY'] = key

        logger.info("[Wizard] Tavily API key written to .env and set in environment")
        return True

    except Exception as e:
        logger.error(f"[Wizard] Failed to write Tavily API key to .env: {e}")
        return False


def validate_tavily_key_format(key: str) -> bool:
    """
    Check if key matches Tavily API key format.

    Args:
        key: API key to validate

    Returns:
        bool: True if format is valid
    """
    key = key.strip()
    # Tavily keys start with 'tvly-' and are reasonably long
    return key.startswith('tvly-') and len(key) > 20


def parse_style_preference(user_input: str) -> str:
    """
    Parse user input into style preference.

    Args:
        user_input: User's response to style question

    Returns:
        str: One of "warm", "balanced", "direct"
    """
    text = user_input.lower().strip()

    # Direct numeric or keyword matches
    if text in ('1', 'warm', 'warm & supportive', 'supportive'):
        return 'warm'
    elif text in ('2', 'balanced', 'default', 'adapt', 'adaptive'):
        return 'balanced'
    elif text in ('3', 'direct', 'concise', 'direct & concise', 'short'):
        return 'direct'

    # Keyword-based inference
    if any(word in text for word in ['empathy', 'supportive', 'warm', 'caring', 'longer']):
        return 'warm'
    if any(word in text for word in ['short', 'brief', 'concise', 'direct', 'terse', 'quick']):
        return 'direct'

    # Default to balanced
    return 'balanced'


def is_skip(user_input: str) -> bool:
    """
    Check if user wants to skip current step.

    Args:
        user_input: User's input

    Returns:
        bool: True if user wants to skip
    """
    text = user_input.lower().strip()
    return text in ('skip', 'none', 'n/a', 'pass', 'no', '-', '')


async def process_wizard_message(
    user_input: str,
    state: WizardState,
    orchestrator
) -> Tuple[str, WizardState, bool]:
    """
    Process a wizard message and return response.

    Args:
        user_input: User's message
        state: Current wizard state
        orchestrator: DaemonOrchestrator instance

    Returns:
        Tuple of (response_text, new_state, is_complete)
    """
    try:
        if state.step == WizardStep.WELCOME:
            return _handle_welcome(user_input, state)

        elif state.step == WizardStep.INTRO:
            return _handle_intro(user_input, state)

        elif state.step == WizardStep.API_KEY:
            return await _handle_api_key(user_input, state, orchestrator)

        elif state.step == WizardStep.TAVILY_KEY:
            return _handle_tavily_key(user_input, state)

        elif state.step == WizardStep.STYLE:
            return _handle_style(user_input, state)

        elif state.step == WizardStep.NAME:
            return _handle_name(user_input, state)

        elif state.step == WizardStep.PRONOUNS:
            return _handle_pronouns(user_input, state)

        elif state.step == WizardStep.BACKGROUND:
            return await _handle_background(user_input, state, orchestrator)

        elif state.step == WizardStep.COMPLETE:
            return "Setup is complete! Please refresh the page to start chatting.", state, True

        else:
            return "Something went wrong. Let's start over.", WizardState(), False

    except Exception as e:
        logger.error(f"[Wizard] Error at step {state.step}: {e}", exc_info=True)
        state.error_count += 1

        if state.error_count >= state.max_retries:
            return (
                "I'm having trouble with setup. You can try restarting the application, "
                "or manually add your API key to the .env file as OPENAI_API_KEY=your-key",
                state,
                False
            )

        return f"Something went wrong. Let's try that step again.", state, False


def _handle_welcome(user_input: str, state: WizardState) -> Tuple[str, WizardState, bool]:
    """Handle welcome step - advance to intro screen."""
    state.step = WizardStep.INTRO
    return get_intro_message(), state, False


def _handle_intro(user_input: str, state: WizardState) -> Tuple[str, WizardState, bool]:
    """Handle intro step - advance to API key collection."""
    state.step = WizardStep.API_KEY
    response = """Now let's get you set up. I'll need an **OpenRouter API key** to connect to language models.

OpenRouter gives you access to Claude, GPT-4, Gemini, and many other models through a single API. You can get a key at **openrouter.ai/keys** (free tier available).

Paste your OpenRouter key below (starts with 'sk-or-')."""
    return response, state, False


async def _handle_api_key(
    user_input: str,
    state: WizardState,
    orchestrator
) -> Tuple[str, WizardState, bool]:
    """
    Handle API key validation and verification.

    Tests the key with a simple API call before accepting.
    """
    key = user_input.strip()

    # Validate format first
    if not validate_api_key_format(key):
        return (
            "That doesn't look like a valid OpenRouter key (should start with 'sk-or-'). "
            "Double-check and try again?",
            state,
            False
        )

    # Test the API key with a simple call
    try:
        logger.info("[Wizard] Testing API key with model call")
        # Set the key in environment AND reinitialize the model manager clients
        os.environ['OPENAI_API_KEY'] = key

        # Reinitialize the model manager's clients with the new API key
        # This is necessary because the ModelManager was created before the API key was available
        if not orchestrator.model_manager.reinitialize_clients(key):
            raise Exception("Failed to initialize API clients")

        test_response = await orchestrator.model_manager.generate_once(
            prompt="Say 'OK' if you can read this.",
            model_name="gpt-4o-mini",
            max_tokens=10,
            temperature=0.0
        )

        if not test_response or not test_response.strip():
            raise Exception("Empty response from API")

        # Check for stub response (indicates API client not properly initialized)
        if test_response.startswith("[API unavailable]"):
            raise Exception("API client not available - got stub response")

        logger.info(f"[Wizard] API key test successful: {test_response[:50]}")

    except Exception as e:
        logger.warning(f"[Wizard] API key validation failed: {e}")
        return (
            "Hmm, that key didn't work — I couldn't connect to the API. "
            "Double-check it and try again?",
            state,
            False
        )

    # Write key to .env
    if not write_api_key_to_env(key):
        # Key works but couldn't save - continue anyway
        logger.warning("[Wizard] API key works but couldn't save to .env")
        return (
            "The key works, but I couldn't save it to .env. "
            "You may need to add it manually. Let's continue anyway.",
            WizardState(step=WizardStep.TAVILY_KEY, collected_data=state.collected_data),
            False
        )

    state.collected_data['api_key_saved'] = True
    state.step = WizardStep.TAVILY_KEY

    response = """Perfect, that's working.

**Optional: Web Search**

I can search the web in real-time when you ask about current events, recent news, or anything that needs up-to-date information. This requires a **Tavily API key**.

You can get one free at **tavily.com** (1000 searches/month on free tier).

Paste your Tavily key below (starts with 'tvly-'), or type **skip** to set this up later."""

    return response, state, False


def _handle_tavily_key(user_input: str, state: WizardState) -> Tuple[str, WizardState, bool]:
    """Handle Tavily API key collection (optional)."""
    text = user_input.strip()

    # Allow skipping
    if is_skip(text):
        logger.info("[Wizard] User skipped Tavily API key setup")
        state.step = WizardStep.STYLE
        response = """No problem — you can add a Tavily key later in your .env file if you want web search.

Now, how would you like me to talk with you?

1. Warm & supportive - More empathetic, longer responses when needed
2. Balanced - Adapts to context (this is the default)
3. Direct & concise - Shorter, to-the-point responses

Just say 1, 2, or 3, or describe what you prefer."""
        return response, state, False

    # Validate format
    if not validate_tavily_key_format(text):
        return (
            "That doesn't look like a valid Tavily key (should start with 'tvly-'). "
            "Double-check and try again, or type **skip** to continue without web search.",
            state,
            False
        )

    # Write key to .env (no API verification - Tavily doesn't have a simple test endpoint)
    if not write_tavily_key_to_env(text):
        logger.warning("[Wizard] Tavily key couldn't be saved to .env")
        return (
            "I couldn't save the key to .env, but let's continue. "
            "You may need to add TAVILY_API_KEY manually.",
            WizardState(step=WizardStep.STYLE, collected_data=state.collected_data),
            False
        )

    state.collected_data['tavily_key_saved'] = True
    state.step = WizardStep.STYLE
    logger.info("[Wizard] Tavily API key saved successfully")

    response = """Got it — web search is now enabled.

How would you like me to talk with you?

1. Warm & supportive - More empathetic, longer responses when needed
2. Balanced - Adapts to context (this is the default)
3. Direct & concise - Shorter, to-the-point responses

Just say 1, 2, or 3, or describe what you prefer."""

    return response, state, False


def _handle_style(user_input: str, state: WizardState) -> Tuple[str, WizardState, bool]:
    """Handle style preference selection."""
    style = parse_style_preference(user_input)
    state.collected_data['style'] = style
    state.step = WizardStep.NAME

    style_names = {
        'warm': 'warm and supportive',
        'balanced': 'balanced',
        'direct': 'direct and concise'
    }
    style_desc = style_names.get(style, style)

    response = f"""Got it, I'll keep things {style_desc}. What should I call you? A name or nickname works — or just say 'skip' if you'd rather not."""

    return response, state, False


def _handle_name(user_input: str, state: WizardState) -> Tuple[str, WizardState, bool]:
    """Handle name collection."""

    if is_skip(user_input):
        state.collected_data['name'] = ''
        state.step = WizardStep.PRONOUNS
        return "No problem. What are your pronouns? (he/him, she/her, they/them, or tell me yours — or skip)", state, False

    name = user_input.strip()
    state.collected_data['name'] = name
    state.step = WizardStep.PRONOUNS

    response = f"Nice to meet you, {name}. What are your pronouns? (he/him, she/her, they/them, or tell me yours — or skip)"

    return response, state, False


def _handle_pronouns(user_input: str, state: WizardState) -> Tuple[str, WizardState, bool]:
    """Handle pronouns collection."""
    if is_skip(user_input):
        state.collected_data['pronouns'] = ''
    else:
        state.collected_data['pronouns'] = user_input.strip()

    state.step = WizardStep.BACKGROUND

    response = """Last thing — is there anything you'd like me to know about you from the start? Could be your work, interests, what you're hoping to use me for... or just skip."""

    return response, state, False


async def _handle_background(
    user_input: str,
    state: WizardState,
    orchestrator
) -> Tuple[str, WizardState, bool]:
    """
    Handle background information collection.

    Extracts facts from user's background text using LLMFactExtractor.
    Separates user facts from entity facts via fact_scope field.
    """
    facts_summary = ""

    if not is_skip(user_input):
        try:
            from memory.llm_fact_extractor import LLMFactExtractor

            logger.info(f"[Wizard] Extracting facts from background text: '{user_input[:100]}...'")
            extractor = LLMFactExtractor(orchestrator.model_manager)
            facts = await extractor.extract_triples([user_input])

            if facts:
                # Separate user facts from entity facts using fact_scope
                user_facts = [f for f in facts if f.get('fact_scope', 'user') == 'user']
                entity_facts = [f for f in facts if f.get('fact_scope') == 'entity']
                state.collected_data['initial_facts'] = user_facts
                state.collected_data['initial_entity_facts'] = entity_facts

                # Summary shows user facts only (most relevant for onboarding display)
                display_facts = user_facts[:3]
                if display_facts:
                    fact_objects = [f.get('object', '') or f.get('value', '') for f in display_facts]
                    facts_summary = f" ({', '.join(fact_objects)}{'...' if len(user_facts) > 3 else ''})"

                logger.info(
                    f"[Wizard] Extracted {len(user_facts)} user facts + "
                    f"{len(entity_facts)} entity facts from background"
                )
            else:
                logger.warning(f"[Wizard] No facts extracted from background text (check LLM Facts logs above)")

        except Exception as e:
            logger.warning(f"[Wizard] Background fact extraction failed: {e}")
            # Continue anyway - not critical

    return await _finalize_wizard(state, orchestrator, facts_summary)


async def _finalize_wizard(
    state: WizardState,
    orchestrator,
    facts_summary: str = ""
) -> Tuple[str, WizardState, bool]:
    """
    Save profile, persist facts to ChromaDB, generate personality file, and complete wizard.

    Stores user facts to both UserProfile (JSON) and ChromaDB (semantic retrieval).
    Entity facts go to ChromaDB only (not UserProfile).
    Generates custom_personality.txt if style != balanced.
    Skips FactVerifier — no existing facts to conflict with on first boot.
    """
    try:
        from memory.user_profile import UserProfile

        # Get or create user profile
        profile = orchestrator.user_profile if hasattr(orchestrator, 'user_profile') else UserProfile()

        if profile is None:
            profile = UserProfile()

        # Update identity
        profile.update_identity(
            name=state.collected_data.get('name', ''),
            pronouns=state.collected_data.get('pronouns', '')
        )

        # Update preferences
        profile.update_preferences(
            style=state.collected_data.get('style', 'balanced')
        )

        # --- Store user facts to UserProfile + ChromaDB ---
        wizard_name = state.collected_data.get('name', '').strip()
        wizard_pronouns = state.collected_data.get('pronouns', '').strip()
        chroma_store = None
        if hasattr(orchestrator, 'memory_system') and orchestrator.memory_system:
            chroma_store = getattr(orchestrator.memory_system, 'chroma_store', None)

        profile_added = 0
        chroma_added = 0
        skipped = 0

        for fact in state.collected_data.get('initial_facts', []):
            relation = fact.get('relation', '')
            obj = fact.get('value') or fact.get('object', '')

            # Skip extracted name/pronouns if wizard already collected them
            if relation == 'name' and wizard_name:
                logger.info(f"[Wizard] Skipping extracted name='{obj}' — wizard collected name='{wizard_name}'")
                skipped += 1
                continue
            if relation == 'pronouns' and wizard_pronouns:
                logger.info(f"[Wizard] Skipping extracted pronouns — wizard collected pronouns='{wizard_pronouns}'")
                skipped += 1
                continue

            # UserProfile (identity/preferences JSON)
            profile.add_fact(
                relation=relation,
                value=obj,
                confidence=fact.get('confidence', 0.7),
                source_excerpt="Initial wizard background",
                category=None  # Auto-categorized
            )
            profile_added += 1

            # ChromaDB (semantic retrieval)
            if chroma_store:
                subj = fact.get('subject', 'user')
                fact_text = f"{subj} | {relation} | {obj}"
                try:
                    chroma_store.add_fact(
                        fact=fact_text,
                        source={
                            "source": "onboarding_wizard",
                            "confidence": fact.get('confidence', 0.7),
                            "fact_scope": "user",
                        },
                    )
                    chroma_added += 1
                except Exception as e:
                    logger.warning(f"[Wizard] ChromaDB add_fact failed: {e}")

        # --- Store entity facts to ChromaDB only (not UserProfile) ---
        entity_added = 0
        for fact in state.collected_data.get('initial_entity_facts', []):
            if not chroma_store:
                break
            subj = fact.get('subject', '')
            relation = fact.get('relation', '')
            obj = fact.get('value') or fact.get('object', '')
            if not (subj and relation and obj):
                continue

            fact_text = f"{subj} | {relation} | {obj}"
            source_dict = {
                "source": "onboarding_wizard",
                "confidence": fact.get('confidence', 0.55),
                "fact_scope": "entity",
            }
            for key in ("entity_type", "user_connection"):
                val = fact.get(key)
                if val:
                    source_dict[key] = val
            try:
                chroma_store.add_fact(fact=fact_text, source=source_dict)
                entity_added += 1
            except Exception as e:
                logger.warning(f"[Wizard] ChromaDB entity add_fact failed: {e}")

        logger.info(
            f"[Wizard] Facts stored: {profile_added} to profile, "
            f"{chroma_added} user + {entity_added} entity to ChromaDB "
            f"(skipped {skipped} conflicting identity facts)"
        )

        # Ensure profile is saved
        profile.save()
        logger.info(
            f"[Wizard] Profile saved. Identity: name='{profile.identity.name}', "
            f"pronouns='{profile.identity.pronouns}', style='{profile.preferences.style}'"
        )

        # Update orchestrator's profile reference
        if hasattr(orchestrator, 'user_profile'):
            orchestrator.user_profile = profile

        # --- Generate personality file from style choice ---
        style = state.collected_data.get('style', 'balanced')
        _apply_personality_style(style)

        logger.info("[Wizard] Wizard complete, orchestrator profile updated")

    except Exception as e:
        logger.error(f"[Wizard] Failed to save profile: {e}", exc_info=True)
        # Continue anyway - wizard completion is more important than save failure

    state.step = WizardStep.COMPLETE

    if facts_summary:
        response = f"Got it — I'll remember that{facts_summary}. We're all set! Feel free to start chatting."
    else:
        response = "Alright, we're all set! Feel free to start chatting whenever you're ready."

    return response, state, True


# ---------------------------------------------------------------------------
# Personality style helpers
# ---------------------------------------------------------------------------

_STYLE_MODIFIERS = {
    'warm': """

## Style Override — Warm & Supportive

Lean into warmth and emotional attunement. When in doubt, choose the more supportive response.
- Give longer, more reflective replies when the user shares something personal
- Offer more affirmation and encouragement (but stay genuine — never hollow)
- Ask follow-up questions that show real interest in how they're feeling
- Default to empathetic framing before practical advice
""",
    'direct': """

## Style Override — Direct & Concise

Prioritize brevity and clarity. Trim anything that doesn't add information.
- Lead with the answer, skip preambles
- Keep replies short unless the topic genuinely requires depth
- Minimize small talk and filler — get to the point
- Default to practical advice before emotional framing
- Still be warm when it matters (emotional support, bad news) — just be efficient about it
""",
}


def _apply_personality_style(style: str) -> None:
    """
    Generate custom_personality.txt from default + style modifier.
    For 'balanced', remove any existing custom file so default is used.
    """
    try:
        from config.app_config import PERSONALITY_CUSTOM_PATH, PERSONALITY_DEFAULT_PATH

        if style == 'balanced':
            # Use default personality — remove custom if it exists
            custom_path = Path(PERSONALITY_CUSTOM_PATH)
            if custom_path.exists():
                custom_path.unlink()
                logger.info("[Wizard] Removed custom personality — using default (balanced)")
            return

        modifier = _STYLE_MODIFIERS.get(style)
        if not modifier:
            logger.warning(f"[Wizard] Unknown style '{style}', skipping personality generation")
            return

        # Read default personality as base
        default_path = Path(PERSONALITY_DEFAULT_PATH)
        if not default_path.exists():
            logger.warning("[Wizard] Default personality file not found, skipping")
            return

        base_text = default_path.read_text(encoding='utf-8')
        custom_text = base_text + modifier

        # Write custom personality
        custom_path = Path(PERSONALITY_CUSTOM_PATH)
        custom_path.parent.mkdir(parents=True, exist_ok=True)
        custom_path.write_text(custom_text, encoding='utf-8')
        logger.info(f"[Wizard] Generated '{style}' personality ({len(custom_text)} chars)")

    except Exception as e:
        logger.warning(f"[Wizard] Personality generation failed (non-critical): {e}")
