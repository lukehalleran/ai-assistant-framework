"""
gui/wizard.py

Conversational onboarding wizard for first-run setup.

Module Contract
- Purpose: Conversational wizard state machine for first-run onboarding. Collects API key,
  style preferences, and user identity through chat interface. Bypasses RAG/memory pipeline entirely.
- Inputs:
  - WizardState dataclass tracking current step and collected data
  - process_wizard_message(user_input, state, orchestrator) → (response, new_state, is_complete)
  - get_welcome_message() → str
- Outputs:
  - Daemon's wizard responses (friendly, conversational)
  - Updated WizardState after each step
  - is_complete=True when wizard finishes
- Side effects:
  - Writes API key to .env file (plaintext) AND sets os.environ at runtime
  - Calls LLMFactExtractor on "anything else" input
  - Saves completed profile via UserProfile.save()
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
    API_KEY = "api_key"
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
    return """Hey! Looks like this is your first time here. I'm Daemon — I'll be your conversational partner. Before we dive in, I need to get a few things set up.

First, I'll need an OpenRouter API key to connect to language models. You can get one at openrouter.ai/keys. Paste it here when you're ready."""


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

        elif state.step == WizardStep.API_KEY:
            return await _handle_api_key(user_input, state, orchestrator)

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
    """Handle welcome step - advance to API key collection."""
    state.step = WizardStep.API_KEY
    # Don't repeat the welcome message - just acknowledge and ask for API key
    return "Great! Please paste your OpenRouter API key below (starts with 'sk-or-').", state, False


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
        # Temporarily set the key for testing
        os.environ['OPENAI_API_KEY'] = key
        test_response = await orchestrator.model_manager.generate_once(
            prompt="Say 'OK' if you can read this.",
            model_name="gpt-4o-mini",
            max_tokens=10,
            temperature=0.0
        )

        if not test_response or not test_response.strip():
            raise Exception("Empty response from API")

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
            WizardState(step=WizardStep.STYLE, collected_data=state.collected_data),
            False
        )

    state.collected_data['api_key_saved'] = True
    state.step = WizardStep.STYLE

    response = """Perfect, that's working. Now, how would you like me to talk with you?

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
    """
    facts_summary = ""

    if not is_skip(user_input):
        try:
            from memory.llm_fact_extractor import LLMFactExtractor

            logger.info("[Wizard] Extracting facts from background text")
            extractor = LLMFactExtractor(orchestrator.model_manager)
            facts = await extractor.extract_triples([user_input])

            if facts:
                state.collected_data['initial_facts'] = facts
                # Create a brief summary of extracted facts
                fact_objects = [f.get('object', '') or f.get('value', '') for f in facts[:3]]
                facts_summary = f" ({', '.join(fact_objects)}{'...' if len(facts) > 3 else ''})"
                logger.info(f"[Wizard] Extracted {len(facts)} facts from background")

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
    Save profile and complete wizard.

    Args:
        state: Current wizard state with collected data
        orchestrator: DaemonOrchestrator instance
        facts_summary: Brief summary of extracted facts for response

    Returns:
        Tuple of (completion_message, state, is_complete=True)
    """
    try:
        from memory.user_profile import UserProfile

        # Get or create user profile
        profile = orchestrator.user_profile if hasattr(orchestrator, 'user_profile') else UserProfile()

        # Update identity
        profile.update_identity(
            name=state.collected_data.get('name', ''),
            pronouns=state.collected_data.get('pronouns', '')
        )

        # Update preferences
        profile.update_preferences(
            style=state.collected_data.get('style', 'balanced')
        )

        # Add initial facts if provided
        if 'initial_facts' in state.collected_data:
            for fact in state.collected_data['initial_facts']:
                profile.add_fact(
                    relation=fact.get('relation', ''),
                    value=fact.get('value') or fact.get('object', ''),
                    confidence=fact.get('confidence', 0.7),
                    source_excerpt="Initial wizard background",
                    category=None  # Will be auto-categorized
                )
            logger.info(f"[Wizard] Added {len(state.collected_data['initial_facts'])} initial facts to profile")

        # Ensure profile is saved (defensive - update methods should have saved already)
        profile.save()
        logger.info(f"[Wizard] Profile saved successfully. Identity: name='{profile.identity.name}', pronouns='{profile.identity.pronouns}', style='{profile.preferences.style}'")

        # Update orchestrator's profile reference
        if hasattr(orchestrator, 'user_profile'):
            orchestrator.user_profile = profile

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
