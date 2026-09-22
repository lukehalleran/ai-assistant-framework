"""tests/smoke/fresh_clone_boot.py

Per-boot script for the isolated-runtime end-to-end smoke contract (Lane 1
Subgoal C, docs/PLAN_20260921_lane1_career_fair_readiness.md). Run as a
subprocess, never imported by the test directly:

    python -s fresh_clone_boot.py --run-dir <dir> --boot {1,2} --report <json>

Proves, against a generic clone with no owner data, the deployed
start -> wizard -> 3 turns -> clean shutdown (real backup) -> restart ->
persisted-state read-back -> clean shutdown contract, using only deployed
functions (main.build_orchestrator/run_shutdown_tasks_async,
utils.single_instance.acquire_single_instance_lock, utils.preflight.run_preflight,
gui.launch.check_first_run, gui.wizard.process_wizard_message,
api.app.create_app, utils.backup_manager). No source/runtime code is
changed by this script; only ModelManager.generate_once/generate_async and
the lifespan's unrelated background workers are stubbed. The real FastAPI
ASGI lifespan and its shutdown/backup hook run.

Boot 1: preflight -> full wizard walk (all steps, incl. the API-key
verification call) -> 3 HTTP turns against a real loopback-bound uvicorn
server -> drain background storage after each turn -> read the turn's
debug record over /api/debug and assert the retrieved-memory context
contains "Pixel" (in-process persist -> retrieve) -> stop the server and
let the production ASGI lifespan run main.run_shutdown_tasks_async (the
ONE deployed shutdown funnel; DAEMON_TEST_MODE is unset in this process,
so the real backup path runs) -> assert a fresh backup archive exists under the run dir ->
release the single-instance lock -> write the JSON report -> exit 0.

Boot 2: same run dir -> check_first_run(orch) is False (persisted profile
identity read back) -> corpus has >= 3 entries and identity.name ==
"Sam Fresh" (read through the deployed UserProfile/CorpusManager objects)
-> one recall turn -> debug record still contains "Pixel" (cross-restart
retrieve) -> shutdown funnel -> backup ran again -> report -> exit 0.

Exit codes: 0 success; 3 isolation-check failure (a resolved path escaped
the clone/run dir); 4 the HF model cache is missing on this machine (the
test treats this as an actionable skip, not a failure); 1 any other
failure (traceback on stderr for diagnosis).
"""

import argparse
import asyncio
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# tests/smoke/fresh_clone_boot.py -> tests/smoke -> tests -> clone root.
CLONE_ROOT = Path(__file__).resolve().parents[2]

WIZARD_NAME = "Sam Fresh"
WIZARD_API_KEY = "sk-or-smoketest0000000000"
PIXEL_TURN_1 = "I adopted a cat named Pixel last week."
PIXEL_TURN_2 = "What are two things to buy for a new cat?"
PIXEL_TURN_3 = "What did I say my cat's name was?"
# boot 2's recall turn is DELIBERATELY reworded, not a verbatim repeat of
# PIXEL_TURN_3 (found during implementation, see the boot-2 finding in this
# batch's report): gui/handlers._recent_completed_duplicate is keyed ONLY on
# normalized query text + a recent corpus timestamp, with no session/process
# scoping. Boot 1's PIXEL_TURN_3 lands in the corpus with a timestamp
# seconds old, so a byte-identical boot-2 turn hits the ingress "resend of a
# just-completed identical turn" shortcut (gui/handlers.py:5728) and is
# served the STORED reply directly -- the entire retrieval/prompt-build/
# debug-record pipeline never runs, so /api/debug has nothing to check. That
# shortcut has no notion of "this is a different process/session" at all, so
# an owner who restarts Daemon and asks the exact same question again within
# _COMPLETED_RESEND_WINDOW_S (300s) would see the same behavior in
# production. This script proves the SAME persist -> retrieve contract with
# an equivalent-but-distinct phrasing instead, since that is what boot 2 is
# actually meant to exercise (see this batch's RETURN "findings").
PIXEL_RECALL_BOOT2 = "Can you remind me what I named my new cat?"
CHAT_SENTINEL = "SMOKE_STUB_SENTINEL_REPLY"

# Sections never blanket-disabled by _build_local_overrides even though they
# carry an "enabled" key — backup is the exact mechanism item 9 of the
# contract needs to prove ran for real (DAEMON_TEST_MODE is unset in this
# process), so it must stay on.
_NEVER_DISABLE = {"backup"}


def _build_local_overrides(run_dir: Path, base_cfg: dict) -> dict:
    """Generic-clone isolation + zero-network config, as gitignored local
    overrides deep-merged over the committed config.yaml (never copied —
    it is already generic; see C_agentB_generic_config_audit.md).

    Blanket enabled:false sweep + light_prompt re-enable + memory/wiki path
    redirection seeded from scripts/audit_runtime_smoke.py's main(); backup
    is excluded from the sweep (see _NEVER_DISABLE) and location/
    personal_vocabulary overrides are added per C_fable_decisions.md
    decisions 7/8.
    """
    overrides: dict = {}
    for name, section in base_cfg.items():
        if name in _NEVER_DISABLE:
            continue
        if isinstance(section, dict) and "enabled" in section:
            overrides[name] = {"enabled": False}

    overrides.setdefault("light_prompt", {})["enabled"] = True
    overrides.setdefault("features", {})["use_stm_pass"] = False
    overrides.setdefault("obsidian", {})["vault_path"] = str(run_dir / "vault")
    overrides.setdefault("memory", {}).update(
        {
            "corpus_file": str(run_dir / "data" / "corpus.json"),
            "chroma_path": str(run_dir / "data" / "chroma"),
            # Wiki retrieval is controlled by numeric limits, not an enabled
            # flag; a timed-out FAISS load otherwise keeps its worker alive.
            "prompt_max_semantic": 0,
            "prompt_max_wiki": 0,
        }
    )
    # Defense in depth (decision 7/8): location.enabled is already False via
    # the sweep above (get_location() returns None before ever reading this),
    # but set it anyway in case that gate ever changes.
    overrides.setdefault("location", {})["override"] = "Testville"
    overrides.setdefault("user_profile", {})["personal_vocabulary"] = {}
    return overrides


def _prepare_run_dir(run_dir: Path) -> None:
    """Write run_dir/config.local.yaml BEFORE any config import.

    config.yaml itself is NOT copied: it is already generic (confirmed by
    C_agentB_generic_config_audit.md section F), so leaving it unwritten and
    letting _candidate_config_paths fall through to the clone's own
    config/config.yaml is safe and simpler than the alternative of copying
    it. Only config.local.yaml (candidate #1, resolved against CWD) is
    written, which is what closes the leak risk in finding 4 of that audit:
    if it existed and went unwritten, config.local.yaml candidate #2 would
    resolve to config/app_config.py's own directory.
    """
    import yaml  # local import: only needed for this one-shot setup

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)

    base_cfg = yaml.safe_load((CLONE_ROOT / "config" / "config.yaml").read_text())
    overrides = _build_local_overrides(run_dir, base_cfg)
    (run_dir / "config.local.yaml").write_text(yaml.safe_dump(overrides))


def _looks_like_missing_model_cache(exc: BaseException) -> bool:
    text = str(exc)
    markers = (
        "HF_HUB_OFFLINE",
        "offline mode",
        "local_files_only",
        "LocalEntryNotFoundError",
        "does not appear to have a file named",
        "Connection error",
        "We couldn't connect",
        "Cannot find the requested files",
    )
    low = text.lower()
    return any(m.lower() in low for m in markers)


def _isolation_check(run_dir: Path) -> list:
    """Fail hard (exit 3) if any resolved path escapes the clone/run dir."""
    import config.app_config as app_config
    import utils.bootstrap as bootstrap

    problems = []
    run_dir_r = run_dir.resolve()

    def _under_clone(label, path_value):
        p = Path(path_value).resolve()
        if p != CLONE_ROOT and CLONE_ROOT not in p.parents:
            problems.append(f"{label} = {p} (not under clone root {CLONE_ROOT})")

    def _under_run_dir(label, path_value):
        p = Path(path_value).resolve()
        if p != run_dir_r and run_dir_r not in p.parents:
            problems.append(f"{label} = {p} (not under run dir {run_dir_r})")

    _under_clone("utils.bootstrap.__file__", bootstrap.__file__)
    _under_clone("config.app_config.__file__", app_config.__file__)

    _under_run_dir("bootstrap.get_user_profile_path()", bootstrap.get_user_profile_path())
    _under_run_dir("app_config.CHROMA_PATH", app_config.CHROMA_PATH)
    _under_run_dir("app_config.CORPUS_FILE", app_config.CORPUS_FILE)
    _under_run_dir(
        "app_config.KNOWLEDGE_GRAPH_PERSIST_PATH", app_config.KNOWLEDGE_GRAPH_PERSIST_PATH
    )
    _under_run_dir("app_config.BACKUP_DIR", app_config.BACKUP_DIR)
    cfg_candidate = app_config._candidate_config_paths("config.local.yaml")[0]
    _under_run_dir("_candidate_config_paths('config.local.yaml')[0]", cfg_candidate)

    if os.environ.get("DAEMON_TEST_MODE") is not None:
        problems.append(
            f"DAEMON_TEST_MODE is set ({os.environ.get('DAEMON_TEST_MODE')!r}) in the boot subprocess"
        )

    if problems:
        print("ISOLATION CHECK FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(3)

    return [
        str(Path(bootstrap.__file__).resolve()),
        str(Path(app_config.__file__).resolve()),
        str(Path(bootstrap.get_user_profile_path()).resolve()),
        str(Path(app_config.CHROMA_PATH).resolve()),
        str(Path(app_config.CORPUS_FILE).resolve()),
        str(Path(app_config.BACKUP_DIR).resolve()),
    ]


# ---------------------------------------------------------------------------
# Deterministic model stub — keyed by prompt/system_prompt shape.
#
# Real generate_once/generate_async callers reachable on this path (verified
# against the deployed source, C_agentA_pathmap.md section 2 + this batch's
# own reading of memory/shutdown_processor.py + gui/wizard.py):
#   - core/response_generator.py:131 generate_async  -> the streamed reply.
#   - gui/wizard.py _handle_api_key                  -> generate_once, tests
#     the OpenRouter key ("Say 'OK' if you can read this.").
#   - memory/llm_fact_extractor.py extract_triples    -> generate_once,
#     shutdown Phase A fact extraction (config-gated callers -- reflection,
#     thread extraction, behavioral patterns, procedural skills, proposals,
#     daily notes, synthesis, STM -- are all disabled by the config sweep
#     above with this scenario's turn/session shape, so they should never
#     fire; branches are still provided below in case a reachable caller
#     changes shape).
# ---------------------------------------------------------------------------


def _generate_once_stub(captured_once):
    async def generate_once(self, prompt, **kwargs):
        text = prompt if isinstance(prompt, str) else str(prompt)
        system_prompt = str(kwargs.get("system_prompt") or "")
        captured_once.append({"prompt": text, "system_prompt": system_prompt, "kwargs": {
            k: v for k, v in kwargs.items() if k != "system_prompt"
        }})

        # Wizard API-key verification (gui/wizard.py _handle_api_key).
        if text.strip() == "Say 'OK' if you can read this.":
            return "OK"

        # LLM fact extraction (memory/llm_fact_extractor.py _build_prompt) and
        # thread extraction (memory/thread_extractor.py EXTRACTION_PROMPT /
        # RESOLUTION_PROMPT) all want a bare JSON array back.
        if "strict JSON arrays" in system_prompt:
            return "[]"
        if "USER MESSAGES (newest last):" in text and text.rstrip().endswith("JSON:"):
            return "[]"
        if "conversation analyst" in text and "open thread" in text.lower():
            return "[]"

        # STM analyzer (core/stm_analyzer.py) — context-analyzer JSON object.
        if "context analyzer" in system_prompt.lower() or "Return JSON only" in text:
            return (
                '{"topic":"smoke test","user_question":"' + text[:40].replace('"', "'") + '",'
                '"intent":"smoke test","tone":"neutral","reference_type":"new_event",'
                '"temporal_facts":[],"open_threads":[],"constraints":[]}'
            )

        # Shutdown session reflection (memory/shutdown_processor.py
        # run_shutdown_reflection) — plain prose, sectioned.
        if "SESSION TOPIC" in text:
            return (
                "1) SESSION TOPIC: fresh-clone smoke test\n"
                "2) KEY ENTITIES: Pixel\n"
                "3) WHAT HAPPENED:\n- A smoke-test turn was recorded.\n"
                "4) PATTERNS & INSIGHTS:\n- none"
            )

        # Query rewrite (core/context_pipeline.py _rewrite_query) — disabled
        # by rewrite_timeout_s: 0 in the committed config, but if reached,
        # echoing the query back is a safe, valid response.
        if text.startswith("Rewrite this user query"):
            return text

        # Safe generic fallback for any other classifier-shaped call.
        return '{"topic":"smoke","tone":"neutral","is_heavy_topic":false}'

    return generate_once


def _generate_async_stub(captured_async):
    async def generate_async(self, prompt, **kwargs):
        captured_async.append(prompt if isinstance(prompt, str) else str(prompt))

        async def stream():
            for text in (f"{CHAT_SENTINEL} ", "This is a deterministic smoke-test reply."):
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=None)]
                )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason="stop")]
            )

        return stream()

    return generate_async


# ---------------------------------------------------------------------------
# Wizard walk
# ---------------------------------------------------------------------------

# Scripted answers for the full WELCOME -> ... -> COMPLETE walk (verified
# step-by-step against gui/wizard.py's handlers): personal mode, a
# placeholder OpenRouter-shaped key (verified via the stubbed generate_once
# above), skip Tavily/Wolfram (mode=user so there is no E2B step), balanced
# style, name "Sam Fresh", skip pronouns, skip Obsidian (confirmed), skip
# the wiki index, skip the background-facts prompt.
_WIZARD_ANSWERS = [
    "hi",  # WELCOME -> INTRO
    "hi",  # INTRO -> MODE
    "1",  # MODE -> API_KEY (personal/user)
    WIZARD_API_KEY,  # API_KEY -> TAVILY_KEY
    "skip",  # TAVILY_KEY -> WOLFRAM_KEY
    "skip",  # WOLFRAM_KEY -> STYLE (user mode: no E2B step)
    "2",  # STYLE -> NAME (balanced)
    WIZARD_NAME,  # NAME -> PRONOUNS
    "skip",  # PRONOUNS -> OBSIDIAN
    "skip",  # OBSIDIAN -> OBSIDIAN_CONFIRM_SKIP
    "yes",  # OBSIDIAN_CONFIRM_SKIP -> WIKI_INDEX
    "skip",  # WIKI_INDEX -> BACKGROUND
    "skip",  # BACKGROUND -> COMPLETE
]


async def _run_wizard(orch) -> dict:
    from gui.wizard import WizardState, process_wizard_message

    state = WizardState()
    steps = []
    is_complete = False
    for answer in _WIZARD_ANSWERS:
        step_before = state.step.value
        response, state, is_complete = await process_wizard_message(answer, state, orch)
        steps.append({"step": step_before, "answer": answer, "is_complete": is_complete})
        if is_complete:
            break

    if not is_complete:
        raise RuntimeError(f"Wizard did not reach COMPLETE; stalled at step={state.step}")

    return {"steps": steps, "final_step": state.step.value}


# ---------------------------------------------------------------------------
# HTTP turn helpers
# ---------------------------------------------------------------------------


async def _drain_pending_storage(timeout: float = 30.0) -> None:
    from gui import handlers

    if handlers._pending_storage_tasks:
        await asyncio.wait_for(
            asyncio.gather(*tuple(handlers._pending_storage_tasks), return_exceptions=True),
            timeout,
        )


async def _send_turn(client, text: str) -> dict:
    resp = await client.post("/api/chat", json={"text": text})
    resp.raise_for_status()
    body = resp.text
    if "event: error" in body:
        raise RuntimeError(f"turn {text!r} returned an SSE error event: {body[:2000]}")
    if "event: complete" not in body:
        raise RuntimeError(f"turn {text!r} never completed: {body[:2000]}")
    await _drain_pending_storage()
    return {"text": text, "status_code": resp.status_code, "body_chars": len(body)}


async def _latest_debug_prompt(client) -> str:
    resp = await client.get("/api/debug")
    resp.raise_for_status()
    data = resp.json()
    records = data.get("records") or []
    if not records:
        raise RuntimeError("/api/debug returned no records after a completed turn")
    prompt = records[-1].get("prompt")
    if not isinstance(prompt, str):
        raise RuntimeError(f"/api/debug latest record has no usable 'prompt' field: {records[-1]!r}")
    return prompt


# ---------------------------------------------------------------------------
# Boot bodies
# ---------------------------------------------------------------------------


async def _serve_and_run(orch, run_dir: Path, turn_body):
    """Run the production FastAPI lifespan on a real loopback socket.

    Unrelated startup workers are patched to deterministic no-ops; the
    production lifespan shutdown hook is left intact and must create a backup.
    """
    import httpx
    import uvicorn
    from api.app import create_app
    from api.launch_auth import LAUNCH_TOKEN_HEADER, generate_launch_secret
    from api.routes import files
    from gui import launch
    import main as runtime

    files._UPLOAD_DIR = str(run_dir / "uploads")

    secret = generate_launch_secret()
    startup_seams = []

    def _no_background_start(_orchestrator):
        startup_seams.append("background_tasks")

    def _no_idle_monitor():
        startup_seams.append("idle_monitor")

    shutdown_started = time.time()
    with patch.object(launch, "start_background_tasks", side_effect=_no_background_start), patch.object(
        runtime, "_idle_monitor_thread", side_effect=_no_idle_monitor
    ):
        app = create_app(orch, start_background=True, launch_secret=secret)
        from api.app import mount_admin_and_frontend
        app = mount_admin_and_frontend(app, orch)
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(16)
        port = listener.getsockname()[1]
        task = asyncio.create_task(server.serve(sockets=[listener]))
        origin = f"http://127.0.0.1:{port}"
        try:
            async with asyncio.timeout(45):
                while not server.started:
                    if task.done():
                        await task
                        raise RuntimeError("server exited before startup")
                    await asyncio.sleep(0.05)
            headers = {LAUNCH_TOKEN_HEADER: secret, "Origin": origin}
            async with httpx.AsyncClient(base_url=origin, timeout=60, headers=headers) as client:
                turn_report = await turn_body(client)
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, 90)
            listener.close()

    assert sorted(startup_seams) == ["background_tasks", "idle_monitor"], (
        f"ASGI lifespan startup seams did not both run: {startup_seams!r}"
    )
    backup = _find_fresh_backup(run_dir, shutdown_started)
    assert backup["found"], f"ASGI lifespan shutdown did not create a fresh backup: {backup}"
    return {"turns": turn_report, "startup_seams": startup_seams, "backup": backup}


def _find_fresh_backup(run_dir: Path, since: float) -> dict:
    """Scan run_dir/data/backups for a manifest written at/after `since`.

    BackupResult is not reachable through main.run_shutdown_tasks_async (its
    own try/except swallows the return value internally -- see
    utils/backup_manager.run_backup + main._do_shutdown_async's "6/6" phase),
    so this is the deployed-artifact ("else") branch of the acceptance
    check: a real manifest.json, written by utils.backup_manager.run_backup,
    naming the copied stores.
    """
    backups_dir = run_dir / "data" / "backups"
    if not backups_dir.is_dir():
        return {"found": False, "reason": f"{backups_dir} does not exist"}

    candidates = []
    for child in backups_dir.iterdir():
        manifest_path = child / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            continue
        if manifest_path.stat().st_mtime >= since - 2.0:
            candidates.append((manifest_path.stat().st_mtime, child, manifest))

    if not candidates:
        return {"found": False, "reason": "no manifest.json written since shutdown began"}

    candidates.sort(key=lambda c: c[0])
    _, path, manifest = candidates[-1]
    files_copied = manifest.get("files", [])
    has_profile = any("user_profile" in f for f in files_copied)
    has_corpus = any("corpus" in f for f in files_copied)
    return {
        "found": True,
        "path": str(path),
        "manifest": manifest,
        "has_profile": has_profile,
        "has_corpus": has_corpus,
    }


async def _boot1(orch, run_dir: Path) -> dict:
    from gui.launch import check_first_run

    assert check_first_run(orch) is True, "fresh clone must start in first-run/wizard state"
    wizard_report = await _run_wizard(orch)
    assert check_first_run(orch) is False, "wizard completion must clear first-run state"
    assert orch.user_profile.identity.name == WIZARD_NAME, (
        f"wizard-collected identity.name={orch.user_profile.identity.name!r} != {WIZARD_NAME!r}"
    )

    async def turns(client):
        session = await client.get("/api/session")
        session.raise_for_status()
        assert session.json()["history"] == []

        t1 = await _send_turn(client, PIXEL_TURN_1)
        t2 = await _send_turn(client, PIXEL_TURN_2)
        t3 = await _send_turn(client, PIXEL_TURN_3)

        prompt = await _latest_debug_prompt(client)
        debug_hit = "Pixel" in prompt

        history = (await client.get("/api/session")).json()["history"]
        return {
            "turns": [t1, t2, t3],
            "debug_hit_pixel": debug_hit,
            "history_messages": len(history),
        }

    server_report = await _serve_and_run(orch, run_dir, turns)
    turn_report = server_report["turns"]
    assert turn_report["debug_hit_pixel"], (
        "turn 3's debug record prompt did not contain 'Pixel' -- in-process "
        "persist -> retrieve did not round-trip"
    )

    backup = server_report["backup"]
    assert backup["has_profile"] and backup["has_corpus"], (
        f"backup archive is missing the profile/corpus stores: {backup}"
    )

    return {
        "wizard": wizard_report,
        "turns": turn_report,
        "lifespan_startup_seams": server_report["startup_seams"],
        "backup": backup,
    }


async def _boot2(orch, run_dir: Path) -> dict:
    from gui.launch import check_first_run

    assert check_first_run(orch) is False, (
        "boot 2 must read back boot 1's persisted profile/corpus and NOT re-enter first-run"
    )
    assert orch.user_profile.identity.name == WIZARD_NAME, (
        f"boot 2 identity.name={orch.user_profile.identity.name!r} != {WIZARD_NAME!r}"
    )
    corpus_count = len(orch.memory_system.corpus_manager.corpus)
    assert corpus_count >= 3, f"boot 2 corpus has only {corpus_count} entries, expected >= 3"

    async def turns(client):
        t = await _send_turn(client, PIXEL_RECALL_BOOT2)
        prompt = await _latest_debug_prompt(client)
        debug_hit = "Pixel" in prompt
        return {"turn": t, "debug_hit_pixel": debug_hit}

    server_report = await _serve_and_run(orch, run_dir, turns)
    turn_report = server_report["turns"]
    assert turn_report["debug_hit_pixel"], (
        "boot 2's recall turn debug record did not contain 'Pixel' -- "
        "cross-restart persist -> retrieve did not round-trip"
    )

    backup = server_report["backup"]

    return {
        "corpus_count": corpus_count,
        "identity_name": orch.user_profile.identity.name,
        "turns": turn_report,
        "lifespan_startup_seams": server_report["startup_seams"],
        "backup": backup,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--boot", type=int, choices=(1, 2), required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    report_path = Path(args.report)

    _prepare_run_dir(run_dir)
    os.chdir(run_dir)
    sys.path.insert(0, str(CLONE_ROOT))

    captured_once: list = []
    captured_async: list = []

    original_connect = socket.socket.connect

    def local_connect(sock, address):
        if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1", "localhost"}:
            raise RuntimeError(f"smoke boot blocked a non-loopback connection to {address!r}")
        return original_connect(sock, address)

    report: dict = {
        "boot": args.boot,
        "run_dir": str(run_dir),
        "clone_root": str(CLONE_ROOT),
        "pid": os.getpid(),
    }

    with patch("dotenv.load_dotenv", return_value=False), patch(
        "socket.socket.connect", local_connect
    ):
        import torch

        torch.set_num_threads(2)
        from models.model_manager import ModelManager

        with patch.object(
            ModelManager, "generate_once", _generate_once_stub(captured_once)
        ), patch.object(ModelManager, "generate_async", _generate_async_stub(captured_async)):
            try:
                import main as runtime
            except Exception as e:
                if _looks_like_missing_model_cache(e):
                    print(f"[Boot] Missing cached HF model (offline mode): {e}", file=sys.stderr)
                    sys.exit(4)
                traceback.print_exc()
                sys.exit(1)

            report["resolved_paths"] = _isolation_check(run_dir)

            from utils.preflight import print_preflight, run_preflight
            from utils.single_instance import acquire_single_instance_lock

            lock_fh = acquire_single_instance_lock()
            try:
                preflight = run_preflight()
                print_preflight(preflight)
                assert preflight.ok, f"preflight reported fatal errors: {preflight.fatal}"
                report["preflight_warnings"] = list(preflight.warnings)

                started = time.perf_counter()
                try:
                    orch = runtime.build_orchestrator()
                except Exception as e:
                    if _looks_like_missing_model_cache(e):
                        print(
                            f"[Boot] Missing cached HF model (offline mode): {e}",
                            file=sys.stderr,
                        )
                        sys.exit(4)
                    raise
                report["build_orchestrator_seconds"] = round(time.perf_counter() - started, 3)

                if args.boot == 1:
                    report["result"] = asyncio.run(_boot1(orch, run_dir))
                else:
                    report["result"] = asyncio.run(_boot2(orch, run_dir))
            finally:
                try:
                    lock_fh.close()
                except OSError:
                    pass

    report["captured_once_calls"] = len(captured_once)
    report["captured_async_calls"] = len(captured_async)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"BOOT_REPORT={report_path}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except AssertionError as e:
        print(f"[Boot] Assertion failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
