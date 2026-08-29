"""Wave-1 curators (docs/AUTONOMOUS_CURATION_DESIGN.md).

Every curator applies THE deployed predicate/transform (CLAUDE.md
validation rule — never a re-derivation), only proposes reversible
instruments, and ships sentinel cases whose failure aborts its batch.
"""

from memory.curation.curators.error_sentinels import ErrorSentinelCurator
from memory.curation.curators.junk_facts import JunkFactCurator
from memory.curation.curators.stream_artifacts import StreamArtifactCurator
from memory.curation.curators.temporal_staleness import TemporalStalenessCurator

ALL_CURATORS = [
    ErrorSentinelCurator,
    StreamArtifactCurator,
    JunkFactCurator,
    TemporalStalenessCurator,
]
