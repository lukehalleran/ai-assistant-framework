# agent_branch/__init__.py
"""
Agent Branch Portfolio — supervisor harness (Milestone 1).

Milestone 1 scope: build the supervisor harness and PROVE isolation holds for a
single worker. Not the portfolio, not the GUI, not real data.

Trust model
-----------
- The *supervisor* is trusted. It owns the manifest, the evaluation, the LLM
  proxy, and all external writes.
- The *worker* is untrusted. It runs in a rootless-podman container with
  ``--network=none`` (no IP egress), a read-only manifest mount, a
  write-only-from-worker artifacts mount, fake secrets, and synthetic stores.
- The May 18 in-process guards (``utils/*_guard.py``) are COOPERATIVE — an
  adversarial worker bypasses them. They remain as defense-in-depth + audit
  inside the container, but the isolation proof must hold with them assumed OFF.
  The container/namespace alone is the boundary.

Deferred (do NOT build in M1): the LLM judge (Layer 2), scoring, reuse of the
full agentic controller, and any change to the ProposalStore / GUI / feature
registry. See ``agent_branch.eval_layer2`` / ``agent_branch.scoring`` stubs.
"""
