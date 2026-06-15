# agent_branch/workers/__init__.py
"""
Worker entrypoints that run INSIDE the isolated container.

These scripts are stdlib-only on purpose: they execute in the bare worker image
(``python:3.11-slim``) with no project deps and no credentials. They are copied
to ``/work/worker.py`` by the provisioner and run as ``python /work/worker.py``.

- ``attacker.py`` — scripted adversary; deterministically attempts each escape
  and writes an (untrusted) report. The supervisor proves isolation STRUCTURALLY;
  the report is only corroboration.
- ``benign_llm.py`` — minimal LLM-driven worker; makes one bounded allowed edit
  via the supervisor's UDS proxy (the happy-path smoke test).
"""
