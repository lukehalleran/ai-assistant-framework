"""Curation service — the one place the live daemon (shutdown phase, API
routes) gets a configured CurationEngine.

Running IN-PROCESS is the point: the engine writes through the daemon's own
store objects, so the external-script clobber problem (a live instance
re-saving over a script's writes — the 08-05 profile incident) cannot
happen, and no daemon-running guard is needed.
"""

import threading
from typing import Optional

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
    from config.app_config import (
        CURATION_ANOMALY_FRACTION,
        CURATION_AUTO_RATE_CAP,
        CURATION_CURATOR_MODES,
        CURATION_ENABLED,
        CURATION_MAX_MODE,
        CURATION_MAX_QUEUE_ITEMS_PER_CURATOR,
    )
    if not CURATION_ENABLED:
        return None
    with _lock:
        engine = CurationEngine(
            StoreBundle(
                chroma_store=chroma_store,
                user_profile=user_profile,
                corpus_manager=corpus_manager,
                graph_memory=graph_memory,
            ),
            max_mode=CURATION_MAX_MODE,
            curator_modes=CURATION_CURATOR_MODES,
            auto_rate_cap=CURATION_AUTO_RATE_CAP,
            anomaly_fraction=CURATION_ANOMALY_FRACTION,
            max_queue_items_per_curator=CURATION_MAX_QUEUE_ITEMS_PER_CURATOR,
        )
        from config.app_config import CURATION_STALENESS_GRACE_HOURS
        from memory.curation.curators import (
            ErrorSentinelCurator,
            GraphTemporalNodeCurator,
            JunkFactCurator,
            ProfileJunkFactCurator,
            StreamArtifactCurator,
            TemporalStalenessCurator,
        )
        for curator in (
            ErrorSentinelCurator(),
            StreamArtifactCurator(),
            JunkFactCurator(),
            TemporalStalenessCurator(grace_hours=CURATION_STALENESS_GRACE_HOURS),
            ProfileJunkFactCurator(),
            GraphTemporalNodeCurator(),
        ):
            try:
                engine.register(curator)
            except Exception as e:
                logger.warning(f"[Curation] curator registration failed: {e}")
        _engine = engine
        return engine


def get_engine() -> Optional[CurationEngine]:
    return _engine
