"""
core/escalation_tracker.py

Session-level emotional momentum tracker for adaptive tone de-escalation.

Sits between ToneDetector output and prompt building to track:
- Consecutive crisis/elevated message count
- Whether the user engaged with previous suggestions or ignored them
- Escalation velocity (how fast tone shifted)

Based on these signals, recommends a ResponseStrategy that modifies
the tone instructions injected into the system prompt. Prevents the
"therapeutic echo chamber" problem where identical validating responses
are repeated when a user is spiraling.

Integration:
    Orchestrator.__init__:  self.escalation_tracker = EscalationTracker()
    process_user_query:     tracker.update(tone_level, user_message)
    build_full_prompt:      append tracker.get_strategy_instructions()
    after generation:       tracker.record_response(response_text)
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from utils.logging_utils import get_logger

logger = get_logger("escalation_tracker")


class ResponseStrategy(Enum):
    """Response strategy based on session emotional momentum."""

    # Default: validate feelings, offer suggestions/perspective
    VALIDATE_AND_SUGGEST = "validate_and_suggest"

    # Sustained escalation: drop advice, focus on acknowledgment
    GROUNDING_PRESENCE = "grounding_presence"

    # Sustained escalation + ignored suggestions: minimal presence only
    QUIET_COMPANIONSHIP = "quiet_companionship"

    # After de-escalation from sustained distress: carefully re-engage
    GENTLE_REENGAGEMENT = "gentle_reengagement"


# Tone levels that count as "elevated" for escalation tracking
# Import here to avoid circular imports at module level
_ELEVATED_LEVELS = None


def _get_elevated_levels():
    """Lazy import of ToneLevel to avoid circular imports."""
    global _ELEVATED_LEVELS
    if _ELEVATED_LEVELS is None:
        from core.context_pipeline import ToneLevel
        _ELEVATED_LEVELS = {ToneLevel.CRISIS, ToneLevel.ELEVATED}
    return _ELEVATED_LEVELS


class EscalationTracker:
    """
    Tracks session-level emotional momentum and recommends response strategies.

    The tracker maintains a sliding window of tone levels and monitors:
    - How many consecutive messages are at ELEVATED or CRISIS level
    - Whether the user engaged with suggestions in previous responses
    - How rapidly the tone is escalating

    Strategy transitions:
        VALIDATE_AND_SUGGEST  (default, < threshold consecutive elevated)
            ↓ (threshold+ consecutive elevated)
        GROUNDING_PRESENCE    (drop suggestions, pure acknowledgment)
            ↓ (2+ ignored suggestions while elevated)
        QUIET_COMPANIONSHIP   (minimal presence, 1-2 sentences max)
            ↓ (tone drops to CONCERN/CONVERSATIONAL)
        GENTLE_REENGAGEMENT   (carefully re-introduce engagement)
            ↓ (sustained calm)
        VALIDATE_AND_SUGGEST  (back to normal)
    """

    def __init__(
        self,
        escalation_threshold: int = 3,
        deescalation_window: int = 2,
        max_history: int = 10,
    ):
        """
        Args:
            escalation_threshold: Consecutive elevated messages before strategy shift
            deescalation_window: Consecutive calm messages before gentle re-engagement ends
            max_history: Sliding window size for tone history
        """
        self.escalation_threshold = escalation_threshold
        self.deescalation_window = deescalation_window
        self.max_history = max_history

        # State
        self.tone_history: List = []  # List[ToneLevel]
        self.consecutive_elevated_count: int = 0
        self.consecutive_calm_count: int = 0
        self.last_suggestions: List[str] = []
        self.ignored_suggestion_count: int = 0
        self.current_strategy: ResponseStrategy = ResponseStrategy.VALIDATE_AND_SUGGEST

        # Need type history for de-escalation nuance
        self._last_need_type: Optional[str] = None

    def update(
        self,
        tone_level,
        user_message: str,
        need_type: Optional[str] = None,
    ) -> ResponseStrategy:
        """
        Update tracker with a new message and return the recommended strategy.

        Args:
            tone_level: ToneLevel from context pipeline
            user_message: The user's message text (for engagement detection)
            need_type: Optional NeedType value string ("PRESENCE", "PERSPECTIVE", "NEUTRAL")

        Returns:
            Recommended ResponseStrategy for this turn
        """
        elevated_levels = _get_elevated_levels()

        # Add to history
        self.tone_history.append(tone_level)
        if len(self.tone_history) > self.max_history:
            self.tone_history = self.tone_history[-self.max_history:]

        is_elevated = tone_level in elevated_levels

        # Track consecutive counts
        if is_elevated:
            self.consecutive_elevated_count += 1
            self.consecutive_calm_count = 0
        else:
            self.consecutive_calm_count += 1
            # Don't reset consecutive_elevated_count immediately —
            # we need it for de-escalation detection
            pass

        # Check engagement with previous suggestions
        if self.last_suggestions and is_elevated:
            engaged = self._detect_engagement(user_message, self.last_suggestions)
            if not engaged:
                self.ignored_suggestion_count += 1
            else:
                self.ignored_suggestion_count = max(0, self.ignored_suggestion_count - 1)
        elif self.last_suggestions and not is_elevated:
            # If they've calmed down and responded, engagement is implicit
            engaged = self._detect_engagement(user_message, self.last_suggestions)
            if engaged:
                self.ignored_suggestion_count = max(0, self.ignored_suggestion_count - 1)

        # Store need type for de-escalation nuance
        self._last_need_type = need_type

        # Compute strategy
        self.current_strategy = self._compute_strategy(tone_level, need_type)

        logger.debug(
            f"[EscalationTracker] tone={tone_level}, "
            f"consecutive_elevated={self.consecutive_elevated_count}, "
            f"ignored_suggestions={self.ignored_suggestion_count}, "
            f"strategy={self.current_strategy.value}"
        )

        # Reset consecutive elevated count AFTER computing strategy
        # so de-escalation detection can use the old count
        if not is_elevated:
            self.consecutive_elevated_count = 0

        return self.current_strategy

    def record_response(self, response: str) -> None:
        """
        Record the assistant's response for engagement detection on the next turn.

        Extracts actionable suggestions from the response text and stores them
        so the next call to update() can check if the user engaged with them.

        Args:
            response: The assistant's response text
        """
        self.last_suggestions = self._extract_suggestions(response)

    def _compute_strategy(
        self,
        current_tone,
        need_type: Optional[str] = None,
    ) -> ResponseStrategy:
        """
        Compute response strategy based on accumulated signals.

        Distinguishes between:
        - Genuine de-escalation (calming down → GENTLE_REENGAGEMENT)
        - Intensity shift (crisis → angry engagement → stay in support mode)

        Transition order is always:
            VALIDATE → GROUNDING → QUIET → (de-escalation) → GENTLE → VALIDATE
        GROUNDING always precedes QUIET to avoid skipping the intermediate step.
        """
        elevated_levels = _get_elevated_levels()
        is_elevated = current_tone in elevated_levels

        # --- De-escalation detection ---
        if not is_elevated and self._was_recently_elevated():
            # Check if the de-escalation window has passed
            if self.consecutive_calm_count > self.deescalation_window:
                return ResponseStrategy.VALIDATE_AND_SUGGEST

            # Distinguish "calming down" from "shifting intensity"
            # If need_type is PERSPECTIVE, user is shifting to analytical mode —
            # they're still processing but want engagement, not quiet presence.
            if need_type == "PERSPECTIVE":
                return ResponseStrategy.VALIDATE_AND_SUGGEST
            return ResponseStrategy.GENTLE_REENGAGEMENT

        # --- Not elevated, no recent escalation ---
        if not is_elevated:
            return ResponseStrategy.VALIDATE_AND_SUGGEST

        # --- Elevated: check severity ---

        # Early escalation — standard support
        if self.consecutive_elevated_count < self.escalation_threshold:
            return ResponseStrategy.VALIDATE_AND_SUGGEST

        # Sustained escalation beyond threshold + suggestions being ignored
        # QUIET only activates AFTER GROUNDING has been tried (consec > threshold)
        if (
            self.consecutive_elevated_count > self.escalation_threshold
            and self.ignored_suggestion_count >= 2
        ):
            return ResponseStrategy.QUIET_COMPANIONSHIP

        # At or past threshold — grounding presence
        return ResponseStrategy.GROUNDING_PRESENCE

    def _was_recently_elevated(self) -> bool:
        """
        Check if any of the recent messages (before the current one) were elevated.

        Looks at the last few messages excluding the current one to detect
        a transition from elevated to calm.
        """
        if len(self.tone_history) < 2:
            return False

        elevated_levels = _get_elevated_levels()
        # Check the messages before the current one
        lookback = self.tone_history[-(self.escalation_threshold + 1):-1]
        return any(t in elevated_levels for t in lookback)

    def _detect_engagement(self, message: str, suggestions: List[str]) -> bool:
        """
        Detect whether the user engaged with previous suggestions.

        Uses simple heuristics:
        1. Explicit engagement phrases ("I tried", "that helped", etc.)
        2. Keyword overlap with suggestion content (filtering stop words)

        Args:
            message: User's current message
            suggestions: List of suggestion strings from previous response

        Returns:
            True if engagement detected
        """
        msg_lower = message.lower()

        # Positive engagement phrases
        engagement_phrases = [
            "i will", "i'll try", "i tried", "i did",
            "that helped", "good idea", "you're right",
            "thank you", "thanks for", "i appreciate",
            "i went", "i called", "i talked to",
            "i'm going to", "gonna try", "might try",
            "that makes sense", "fair point",
        ]
        for phrase in engagement_phrases:
            if phrase in msg_lower:
                return True

        # Check for keyword overlap with suggestion content
        stop_words = {
            "the", "a", "an", "to", "is", "it", "i", "and", "or", "in",
            "of", "for", "you", "your", "that", "this", "can", "with",
            "be", "do", "if", "on", "at", "by", "so", "not", "but",
            "are", "was", "were", "have", "has", "had", "just", "about",
        }
        message_words = set(msg_lower.split()) - stop_words

        for suggestion in suggestions:
            suggestion_words = set(suggestion.lower().split()) - stop_words
            overlap = suggestion_words & message_words
            if len(overlap) >= 2:
                return True

        return False

    def _extract_suggestions(self, response: str) -> List[str]:
        """
        Extract actionable suggestions from the assistant's response.

        Looks for common suggestion patterns like "Try...", "Consider...",
        "You might...", bullet points with suggestion keywords, etc.
        Handles suggestions that appear mid-line after sentence boundaries.

        Args:
            response: Assistant's response text

        Returns:
            List of extracted suggestion strings
        """
        import re

        suggestions = []

        suggestion_starters = (
            "try ", "consider ", "you might ", "you could ",
            "maybe ", "how about ", "have you ", "it might help",
            "one thing ", "something that ", "perhaps ",
            "i'd suggest ", "i suggest ", "what if ",
        )
        suggestion_keywords = {
            "try", "consider", "might", "could", "suggest",
            "recommend", "helpful", "help", "maybe",
        }

        # Split by newlines, then by sentence boundaries within each line
        for line in response.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue

            # Bullet point suggestions (check before sentence splitting)
            if stripped.startswith(('-', '*')) and any(
                w in stripped.lower() for w in suggestion_keywords
            ):
                suggestions.append(stripped.lstrip('-*  '))
                continue

            # Split line into sentences at period/exclamation/question + space
            sentences = re.split(r'(?<=[.!?])\s+', stripped)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if sentence.lower().startswith(suggestion_starters):
                    suggestions.append(sentence)

        return suggestions

    def get_strategy_instructions(self) -> str:
        """
        Get supplemental tone instructions for the current strategy.

        These instructions are appended to the system prompt AFTER the
        standard tone instructions. They override/augment the base
        response mode when the tracker detects sustained escalation.

        Returns:
            Strategy-specific instruction string, or empty string for default
        """
        if self.current_strategy == ResponseStrategy.GROUNDING_PRESENCE:
            return (
                "\n\n## ESCALATION ADAPTATION: GROUNDING PRESENCE\n"
                "The user has been in sustained emotional distress across multiple messages. "
                "Previous supportive approaches have not shifted the pattern.\n"
                "- Maximum 2-3 sentences. No advice, no suggestions.\n"
                "- Pure acknowledgment: 'I hear you. This is really hard.'\n"
                "- Do NOT repeat coping suggestions already given.\n"
                "- Match their intensity with presence, not words.\n"
                "- If they're venting, let them vent without redirecting.\n"
                "- Silence and brevity can be more powerful than paragraphs."
            )
        elif self.current_strategy == ResponseStrategy.QUIET_COMPANIONSHIP:
            return (
                "\n\n## ESCALATION ADAPTATION: QUIET COMPANIONSHIP\n"
                "The user is in sustained distress and has not engaged with suggestions. "
                "They need presence, not advice.\n"
                "- Maximum 1-2 sentences. Absolute minimum.\n"
                "- Just be there: 'I'm here.' or 'I'm listening.'\n"
                "- No suggestions, no reframes, no coping strategies.\n"
                "- Don't try to be helpful -- just be present.\n"
                "- Less is more. The relationship itself is the support."
            )
        elif self.current_strategy == ResponseStrategy.GENTLE_REENGAGEMENT:
            return (
                "\n\n## ESCALATION ADAPTATION: GENTLE REENGAGEMENT\n"
                "The user was recently in sustained distress but their tone has shifted. "
                "Gently re-engage without rushing.\n"
                "- Acknowledge the shift warmly but don't draw attention to it.\n"
                "- Keep responses moderate length (2-4 sentences).\n"
                "- You may carefully offer ONE small, concrete suggestion.\n"
                "- Don't reference the previous distress unless they bring it up.\n"
                "- Let them set the pace for returning to normal conversation."
            )

        # VALIDATE_AND_SUGGEST: no override, use standard tone instructions
        return ""

    def get_token_budget_override(self) -> Optional[int]:
        """
        Get a token budget override based on the current strategy.

        Shorter strategies need fewer tokens to prevent the model from
        over-generating when instructions say "be brief."

        Returns:
            Token budget override, or None to use the default tone-based budget
        """
        if self.current_strategy == ResponseStrategy.QUIET_COMPANIONSHIP:
            return 300  # Very short: 1-2 sentences
        elif self.current_strategy == ResponseStrategy.GROUNDING_PRESENCE:
            return 500  # Short: 2-3 sentences
        elif self.current_strategy == ResponseStrategy.GENTLE_REENGAGEMENT:
            return 800  # Moderate: 2-4 sentences
        return None  # Use default tone-based budget

    def get_escalation_velocity(self) -> float:
        """
        Calculate how quickly tone has been escalating.

        Returns:
            Float from 0.0 (stable/descending) to 1.0 (rapid escalation)
        """
        if len(self.tone_history) < 2:
            return 0.0

        from core.context_pipeline import ToneLevel
        tone_values = {
            ToneLevel.CONVERSATIONAL: 0,
            ToneLevel.CONCERN: 1,
            ToneLevel.ELEVATED: 2,
            ToneLevel.CRISIS: 3,
        }

        recent = self.tone_history[-5:] if len(self.tone_history) >= 5 else self.tone_history
        values = [tone_values.get(t, 0) for t in recent]

        # Calculate average step change
        deltas = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        if not deltas:
            return 0.0

        avg_delta = sum(deltas) / len(deltas)
        # Normalize to 0-1 range (max possible delta per step is 3)
        return max(0.0, min(1.0, avg_delta / 3.0))

    def get_debug_info(self) -> Dict[str, Any]:
        """Get tracker state for debug_info logging."""
        return {
            "strategy": self.current_strategy.value,
            "consecutive_elevated": self.consecutive_elevated_count,
            "consecutive_calm": self.consecutive_calm_count,
            "ignored_suggestions": self.ignored_suggestion_count,
            "velocity": self.get_escalation_velocity(),
            "history_length": len(self.tone_history),
            "last_suggestions_count": len(self.last_suggestions),
        }

    def reset(self) -> None:
        """Reset tracker state (e.g., new session)."""
        self.tone_history.clear()
        self.consecutive_elevated_count = 0
        self.consecutive_calm_count = 0
        self.last_suggestions.clear()
        self.ignored_suggestion_count = 0
        self.current_strategy = ResponseStrategy.VALIDATE_AND_SUGGEST
        self._last_need_type = None
