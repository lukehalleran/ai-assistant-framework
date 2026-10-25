"""Startup-abort error for an unreadable config file (2026-10-25, class: BC-58/BC-82).

Before this, a syntax error in config.yaml or config.local.yaml logged a warning and
loaded ``{}``: every override (and, for config.yaml, every deliberate OFF decision)
silently fell back to the app_config literals. Same doctrine as
``utils.safe_json.CorruptStoreError``: refuse to start rather than run on a guess.
A MISSING config.local.yaml is fine (it is optional); a PRESENT-but-broken one is not.
"""
from __future__ import annotations


class ConfigParseError(RuntimeError):
    """An existing config file could not be parsed into a mapping."""

    def __init__(self, path, label: str, original):
        self.path = str(path)
        self.label = label
        self.original = original
        super().__init__(
            f"{label} at '{self.path}' exists but could not be parsed ({original}). "
            f"Refusing to start: every setting in it would silently fall back to a built-in "
            f"default. Fix the YAML syntax (a YAML-aware editor or "
            f"`python -c \"import yaml,sys; yaml.safe_load(open(sys.argv[1]))\" '{self.path}'` "
            f"shows the line), or move the file aside to start without it deliberately."
        )
