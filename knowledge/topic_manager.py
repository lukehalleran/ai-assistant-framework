# knowledge/topic_manager.py
import logging

# Use the root logger or create a child logger that will inherit handlers
logger = logging.getLogger(__name__)
logger.debug("topic_manager.py is alive")
import json
from collections import Counter
import spacy
import os

class TopicManager:
    def __init__(self, top_topics_file="data/top_topics.json"):
        self.top_topics_file = top_topics_file
        self.default_common_topics = [
            "World War II", "United States", "Artificial Intelligence",
            "Climate Change", "Photosynthesis", "Quantum Mechanics",
            "Neural Networks", "Moon Landing", "French Revolution",
        ]
        self.user_topic_counter = Counter()
        self._load_top_topics()
        self.nlp = spacy.load("en_core_web_sm")
        self.last_topic: str | None = None  # <- track most-recent primary topic

    def _load_top_topics(self):
        if os.path.exists(self.top_topics_file):
            with open(self.top_topics_file, "r") as f:
                self.top_topics = set(json.load(f))
        else:
            self.top_topics = set(self.default_common_topics)

    def extract_nouns(self, text):
        doc = self.nlp(text)
        return [token.text.lower() for token in doc if token.pos_ == "NOUN"]

    def extract_entities_and_topics(self, text):
        """Extract named entities and important nouns from text"""
        doc = self.nlp(text)

        # Get named entities (proper nouns, places, etc.)
        entities = [ent.text for ent in doc.ents]

        # Get important nouns (not just any noun)
        important_nouns = []
        for token in doc:
            if (token.pos_ == "NOUN" and
                len(token.text) > 3 and  # Skip short words
                not token.is_stop and    # Skip stop words
                token.is_alpha):         # Skip numbers/punctuation
                important_nouns.append(token.text.title())  # Capitalize

        # Combine and deduplicate
        topics = list(set(entities + important_nouns))
        return topics[:10]  # Return top 10 most relevant

    def update_from_user_input(self, text):
        # Extract topics from current input
        current_topics = self.extract_entities_and_topics(text)

        # Update the counter for long-term tracking
        nouns = self.extract_nouns(text)
        self.user_topic_counter.update(nouns)

        # Use current topics as the active topics for this query
        self.top_topics = set(current_topics)

        # Remember “primary” topic for the most recent text
        self.last_topic = current_topics[0] if current_topics else None

        return current_topics

    def get_primary_topic(self, text: str | None = None) -> str | None:
        """
        Return a single best topic string. If text is provided, derive from it;
        else use the last seen topic from update_from_user_input.
        """
        if text:
            topics = self.extract_entities_and_topics(text)
            self.last_topic = topics[0] if topics else None
        return self.last_topic

    def refresh_top_topics(self):
        popular_user_topics = [topic for topic, count in self.user_topic_counter.items() if count >= 3]
        combined = set(self.default_common_topics).union(popular_user_topics)
        with open(self.top_topics_file, "w") as f:
            json.dump(list(combined), f, indent=2)
        print("[TopicManager] Refreshed top topics.")
