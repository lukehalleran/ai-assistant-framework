# memory/memory_scorer.py
from datetime import datetime
from typing import List, Dict

class MemoryScorer:
    def __init__(self):
        pass

    def apply_temporal_decay(self, memories: List[Dict]) -> List[Dict]:
        """Apply temporal decay to memory scores"""
        now = datetime.now()

        for mem_dict in memories:
            memory = mem_dict['memory']

            # Calculate age in hours
            age_hours = (now - memory.timestamp).total_seconds() / 3600.0
            decay_factor = 1.0 / (1.0 + memory.decay_rate * (age_hours/24.0))

            # Boost recently accessed memories
            access_recency = (now - memory.last_accessed).days
            access_boost = 1.0 if access_recency < 1 else 1.0 / (1.0 + 0.1 * access_recency)

            # Calculate final score
            truth = getattr(memory, 'truth_score', memory.metadata.get('truth_score', 0.5))
            mem_dict['final_score'] = (
                mem_dict['relevance_score'] *
                max(0.1, memory.importance_score) *
                max(0.1, decay_factor) *
                (0.75 + 0.5*truth)
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
