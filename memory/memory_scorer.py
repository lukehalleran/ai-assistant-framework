# memory/memory_scorer.py
from datetime import datetime
from typing import List, Dict, Optional

class MemoryScorer:
    def __init__(self, time_manager=None):
        """
        Initialize MemoryScorer with optional time_manager for active day decay.

        Args:
            time_manager: Optional TimeManager instance for active day decay calculation.
                         If None, falls back to hourly decay.
        """
        self.time_manager = time_manager

    def apply_temporal_decay(self, memories: List[Dict]) -> List[Dict]:
        """Apply temporal decay to memory scores"""
        now = datetime.now()

        for mem_dict in memories:
            # Handle flat dictionary structure (both reflections and conversations)
            timestamp = mem_dict.get('timestamp') or mem_dict.get('metadata', {}).get('timestamp')
            if isinstance(timestamp, str):
                try:
                    from dateutil import parser
                    timestamp = parser.parse(timestamp)
                except:
                    # If parsing fails, skip this memory
                    continue

            # Get decay rate with fallback
            decay_rate = mem_dict.get('metadata', {}).get('decay_rate', 0.01)

            # Get importance score with fallback
            importance_score = mem_dict.get('importance_score', 0.5)

            # Get truth score with fallback
            truth_score = mem_dict.get('truth_score', mem_dict.get('metadata', {}).get('truth_score', 0.5))

            # Use active day decay if time_manager supports it, otherwise fallback to hourly decay
            if (self.time_manager is not None and
                hasattr(self.time_manager, 'calculate_active_day_decay')):
                decay_factor = self.time_manager.calculate_active_day_decay(timestamp, decay_rate)
            else:
                # Fallback to original hourly decay (less aggressive)
                age_hours = (now - timestamp).total_seconds() / 3600.0
                decay_factor = 1.0 / (1.0 + decay_rate * (age_hours/168.0))  # Weekly instead of daily decay

            # Calculate final score
            mem_dict['final_score'] = (
                mem_dict.get('relevance_score', 0.0) *
                max(0.1, importance_score) *
                max(0.1, decay_factor) *
                (0.75 + 0.5*truth_score)
            )

        return memories

    def calculate_importance_score(self, content: str) -> float:
        """Calculate importance score using simple heuristics"""
        score = 0.5

        # Boost for longer content
        if len(content) > 200:
            score += 0.1

        # Boost for questions
        if '?' in content:
            score += 0.1

        # Boost for certain keywords
        important_keywords = ['important', 'remember', 'note', 'key', 'critical', 'essential']
        if any(kw in content.lower() for kw in important_keywords):
            score += 0.2

        return min(score, 1.0)
