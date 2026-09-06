"""Wave-1 curators (docs/AUTONOMOUS_CURATION_DESIGN.md §3 — deterministic
ports of the terminal cleanup scripts) plus the Wave-2 TemporalStaleness
curator. The 2026-09-05 additions (ProfileJunkFactCurator,
GraphTemporalNodeCurator) are Wave-1 class: deployed predicates, reversible
instruments.

Every curator applies THE deployed predicate/transform (CLAUDE.md
validation rule — never a re-derivation), only proposes reversible
instruments, and ships sentinel cases whose failure aborts its batch.
"""

from memory.curation.curators.error_sentinels import ErrorSentinelCurator
from memory.curation.curators.graph_temporal_nodes import GraphTemporalNodeCurator
from memory.curation.curators.junk_facts import JunkFactCurator
from memory.curation.curators.profile_junk_facts import ProfileJunkFactCurator
from memory.curation.curators.stream_artifacts import StreamArtifactCurator
from memory.curation.curators.temporal_staleness import TemporalStalenessCurator

ALL_CURATORS = [
    ErrorSentinelCurator,
    StreamArtifactCurator,
    JunkFactCurator,
    TemporalStalenessCurator,
    ProfileJunkFactCurator,
    GraphTemporalNodeCurator,
]
