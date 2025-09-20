# core/dependencies.py
"""Centralized dependency management to prevent circular imports"""


class DependencyContainer:
    """Singleton container for shared dependencies"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self, model_manager):
        """Initialize all shared dependencies once"""
        if self._initialized:
            return

        from models.tokenizer_manager import TokenizerManager

        self.model_manager = model_manager
        self.tokenizer_manager = TokenizerManager(model_manager=model_manager)
        self._initialized = True

    def get_tokenizer_manager(self):
        if not self._initialized:
            raise RuntimeError("Dependencies not initialized. Call initialize() first.")
        return self.tokenizer_manager

    def get_model_manager(self):
        if not self._initialized:
            raise RuntimeError("Dependencies not initialized. Call initialize() first.")
        return self.model_manager

# Global instance
deps = DependencyContainer()
