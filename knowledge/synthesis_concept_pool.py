"""
# knowledge/synthesis_concept_pool.py

Module Contract
- Purpose: The curated pool of PROMINENT cross-domain concepts that the discovery
  ("scientific insight") synthesis generator pairs from. Externalized here (out of the
  scripts/ experiment tree) as the single production source of the discovery substrate.
- Exports: CONCEPT_POOL (list[str]) — prominent, well-developed scientific/technical
  concepts spanning physics, biology, math/CS, economics, control/network theory.
- Why prominence matters (validated 2026-06-30, scripts/validate_anchored_generator.py):
  through the REAL coherence judge, pairing two PROMINENT concepts in the non-obvious
  band (cos 0.2-0.45) yields ~46% MODERATE+STRONG / ~17% accept, vs ~0 for the old
  personal->wiki generators (which paired low-prominence personal entities) and vs 31%
  for FAISS-neighbour "anchored" pairing (its neighbourhoods drag in obscure titles like
  a song or a person's name). The lever is concept QUALITY, not anchoring or graph walks.
- Growable: append prominent concepts (one well-defined idea per entry, lowercase except
  proper nouns). Avoid narrow/obscure titles and avoid near-synonyms of existing entries.
- Consumed by: knowledge/synthesis_pooled_generator.py
"""

from __future__ import annotations

# Curated prominent cross-domain concepts (union of the validated discovery POOL +
# ANCHORS used in the anchored-vs-random PoC; deduped, order-stable).
CONCEPT_POOL: list[str] = [
    # physics / complex systems
    "entropy",
    "thermodynamics",
    "statistical mechanics",
    "phase transition",
    "percolation theory",
    "renormalization group",
    "resonance",
    "oscillation",
    "diffusion",
    "turbulence",
    "chaos theory",
    "fractal",
    "crystallography",
    "metallurgy",
    "symmetry",
    "equilibrium",
    # biology / life
    "natural selection",
    "evolution",
    "epidemic",
    "metabolism",
    "homeostasis",
    "morphogenesis",
    "ecology",
    "immune system",
    "photosynthesis",
    "enzyme",
    # math / CS / information
    "information theory",
    "optimization",
    "gradient descent",
    "simulated annealing",
    "neural network",
    "Markov chain",
    "cryptography",
    "error correction",
    "error detection and correction",
    "recursion",
    "topology",
    "control theory",
    "queueing theory",
    "network theory",
    "network science",
    "social network",
    # economics / strategy
    "game theory",
    "supply and demand",
    "auction theory",
    "tragedy of the commons",
    "feedback loop",
    "linguistics",
]
