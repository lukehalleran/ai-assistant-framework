"""Curation service — the one place the live daemon (shutdown phase, API
routes) gets a configured CurationEngine.

Running IN-PROCESS is the point: the engine writes through the daemon's own
store objects, so the external-script clobber problem (a live instance
re-saving over a script's writes — the 08-05 profile incident) cannot
happen, and no daemon-running guard is needed.
"""

import threading
from typing import Optional

from config import app_config
import memory.curation.curators as curators
from memory.curation.engine import CurationEngine, StoreBundle
from utils.logging_utils import get_logger

logger = get_logger("curation_service")

_lock = threading.Lock()
_engine: Optional[CurationEngine] = None


def init_engine(*, chroma_store=None, user_profile=None,
                corpus_manager=None, graph_memory=None) -> Optional[CurationEngine]:
    """Build (or rebuild) the singleton engine from config + live stores.
    Returns None when curation is disabled."""
    global _engine
    if not app_config.CURATION_ENABLED:
        return None
    with _lock:
        engine = CurationEngine(
            StoreBundle(
                chroma_store=chroma_store,
                user_profile=user_profile,
                corpus_manager=corpus_manager,
                graph_memory=graph_memory,
            ),
            max_mode=app_config.CURATION_MAX_MODE,
            curator_modes=app_config.CURATION_CURATOR_MODES,
            auto_rate_cap=app_config.CURATION_AUTO_RATE_CAP,
            anomaly_fraction=app_config.CURATION_ANOMALY_FRACTION,
            max_queue_items_per_curator=app_config.CURATION_MAX_QUEUE_ITEMS_PER_CURATOR,
        )
        for curator in (
            curators.ErrorSentinelCurator(),
            curators.StreamArtifactCurator(),
            curators.JunkFactCurator(),
            curators.TemporalStalenessCurator(grace_hours=app_config.CURATION_STALENESS_GRACE_HOURS),
            curators.ProfileJunkFactCurator(),
            curators.GraphTemporalNodeCurator(),
        ):
            try:
                engine.register(curator)
            except Exception as e:
                logger.warning(f"[Curation] curator registration failed: {e}")
        _engine = engine
        return engine


def get_engine() -> Optional[CurationEngine]:
    return _engine
