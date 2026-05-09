"""
# gui/launch.py

Module Contract
- Purpose: Define and launch the Gradio UI. Assembles tabs (Chat, Debug Trace, Status, Personality
  inline; Proposals, Synthesis, Settings extracted to gui/tabs/). Wires UI events to handlers and state.
  Registers health check endpoint. Routes to wizard for first-run.
- Theme: Extracted to gui/theme.py (DARK_CHATBOT_CSS + get_dark_theme).
- Inputs:
  - Orchestrator instance (built in main). Environment flags for networking.
  - force_wizard: Boolean to force wizard mode for testing
- Outputs:
  - Running web app (local or shared); exposes live chat and debug views.
  - Health check endpoint at /health (HTTP 200 for healthy, 503 for degraded)
- Key pieces:
  - IS_FROZEN: Detects PyInstaller frozen executable mode
  - launch_gui(): Main entry point, routes to wizard or normal UI
  - _launch_wizard_ui(): First-run setup wizard interface
  - _run_daily_notes_catchup(): Background thread for daily notes catch-up on startup [NEW 2026-01-18]
    - Generates yesterday's daily note if missing
    - Non-blocking (runs in daemon thread)
    - Respects DAILY_NOTES_ENABLED config flag
  - _run_reference_docs_seed(): Background thread for auto-seeding docs/ into reference_docs ChromaDB
    - Syncs docs/ directory using file mtime for idempotency (skips unchanged files)
    - Non-blocking (runs in daemon thread)
    - Respects REFERENCE_DOCS_ENABLED and REFERENCE_DOCS_AUTO_SEED config flags
  - submit_chat(): async driver that streams tokens + timer updates
  - Tabs:
    • Chat: chat UI, file upload, raw toggle, Sync Notes button
    • Debug Trace: renders debug_state entries
    • Status: shows counters (summaries, corpus counts, logs)
    • Synthesis: blind review queue with two-layer grading (3 binary screening + 1-5 gut-feel slider); see docs/grading_plan.md
    • Proposals: browse, filter, and generate code proposals from ChromaDB
    • Provenance: per-turn audit trail (session_id, response_mode, model, thinking block, citations, agentic rounds) [RENAMED from Citations 2026-03-26]
    • Settings: slider to control summary cadence (every N exchanges)
    • Personality: live personality text editor with Set/Restore Default buttons [NEW 2026-03-26]
      - Loads current personality from custom_personality.txt or default_personality.txt
      - Set button saves to custom_personality.txt, enforces PERSONALITY_MAX_CHARS limit
      - Restore Default deletes custom file, reloads default
      - Operating principles appended automatically (not editable from GUI)
  - Health endpoint: Lightweight checks (corpus, ChromaDB, API key) for Docker/K8s
- Dependencies:
  - gui.handlers (handle_submit)
  - gui.wizard (WizardState, process_wizard_message)
  - utils.health_check (add_health_endpoint)
  - utils.conversation_logger
  - utils.daily_notes_generator (DailyNotesGenerator) [NEW 2026-01-18]
  - knowledge.reference_docs_manager (ReferenceDocsManager) — auto-seed on startup
- Side effects:
  - Registers /health endpoint on FastAPI app after launch
  - Logs URLs and health endpoint to console
  - [FROZEN MODE] Auto-opens browser via platform-specific command (xdg-open/open/startfile)
  - Runs indefinite event loop (until KeyboardInterrupt)
  - [STARTUP] Spawns background thread for daily notes catch-up [NEW 2026-01-18]
  - [STARTUP] Spawns background thread for reference docs auto-seed (mtime idempotent)
- Threading/Async: Main thread blocked by event loop; Gradio handles async internally.
  Daily notes catch-up and reference docs seed run in separate daemon threads.
"""
import os
import sys
import logging
import socket
import gradio as gr
from gradio import themes
import copy
from gui.handlers import handle_submit
from utils.conversation_logger import get_conversation_logger
from gui.wizard import WizardState, process_wizard_message, get_welcome_message
from gui.theme import DARK_CHATBOT_CSS, get_dark_theme

# Detect frozen executable mode
IS_FROZEN = getattr(sys, 'frozen', False)

def _env_flag(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}

def _find_free_port(preferred: int = 7860) -> int:
    # Try preferred first; if taken, find an ephemeral free port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def _run_daily_notes_catchup():
    """
    Run daily notes catch-up in background thread.
    Generates yesterday's note if missing, plus today's if there are conversations.
    Non-blocking - errors are logged but don't affect GUI startup.

    Note: Creates its own ModelManager to avoid sharing httpx clients across event loops,
    which causes 'Event loop is closed' errors.
    """
    import threading
    import asyncio

    def _catchup_task():
        try:
            from config.app_config import DAILY_NOTES_ENABLED
            if not DAILY_NOTES_ENABLED:
                return

            from utils.daily_notes_generator import DailyNotesGenerator
            from datetime import date, timedelta

            # Create a fresh generator for this thread - don't share model_manager
            # across event loops (httpx.AsyncClient is not thread-safe)
            generator = DailyNotesGenerator()

            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Generate yesterday's note if missing
                result = loop.run_until_complete(generator.generate_yesterday_if_missing())
                if result and result.success:
                    print(f"[DailyNotes] Catch-up: Generated yesterday's note ({result.conversation_count} conversations)")
                elif result and result.skipped_reason:
                    print(f"[DailyNotes] Catch-up: Skipped yesterday ({result.skipped_reason})")
                elif result and result.error:
                    print(f"[DailyNotes] Catch-up FAILED: {result.error}")
            finally:
                loop.close()

        except Exception as e:
            # Non-critical: log and continue
            print(f"[DailyNotes] Catch-up failed (non-critical): {e}")

    # Run in background thread so it doesn't block GUI startup
    thread = threading.Thread(target=_catchup_task, daemon=True)
    thread.start()
    print("[DailyNotes] Started background catch-up check...")


def _run_weekly_notes_catchup():
    """
    Run weekly notes catch-up in background thread.
    Generates last week's summary if the week is complete (today is Monday+).
    Non-blocking - errors are logged but don't affect GUI startup.

    Note: Creates its own ModelManager to avoid sharing httpx clients across event loops,
    which causes 'Event loop is closed' errors.
    """
    import threading
    import asyncio

    def _catchup_task():
        try:
            from config.app_config import WEEKLY_NOTES_ENABLED
            if not WEEKLY_NOTES_ENABLED:
                return

            from utils.weekly_notes_generator import WeeklyNotesGenerator

            # Create a fresh generator for this thread - don't share model_manager
            # across event loops (httpx.AsyncClient is not thread-safe)
            generator = WeeklyNotesGenerator()

            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Generate last week's summary if complete
                result = loop.run_until_complete(generator.generate_last_week_if_complete())
                if result and result.success:
                    print(f"[WeeklyNotes] Catch-up: Generated Week {result.week_num} summary ({result.daily_notes_found} daily notes)")
                elif result and result.skipped_reason:
                    print(f"[WeeklyNotes] Catch-up: Skipped ({result.skipped_reason})")
            finally:
                loop.close()

        except Exception as e:
            # Non-critical: log and continue
            print(f"[WeeklyNotes] Catch-up failed (non-critical): {e}")

    # Run in background thread so it doesn't block GUI startup
    thread = threading.Thread(target=_catchup_task, daemon=True)
    thread.start()
    print("[WeeklyNotes] Started background catch-up check...")


def _run_monthly_notes_catchup():
    """
    Run monthly notes catch-up in background thread.
    Step 1: Migrate legacy weekly folders into monthly parents (sync).
    Step 2: Generate last month's summary if complete and missing (async).
    Non-blocking - errors are logged but don't affect GUI startup.
    """
    import threading
    import asyncio

    def _catchup_task():
        try:
            from config.app_config import MONTHLY_NOTES_ENABLED
            if not MONTHLY_NOTES_ENABLED:
                return

            from utils.monthly_notes_generator import MonthlyNotesGenerator

            generator = MonthlyNotesGenerator()

            # Step 1: Migrate weekly folders to monthly parents (sync, no event loop)
            migrated = generator.migrate_weekly_folders_to_monthly()
            if migrated > 0:
                print(f"[MonthlyNotes] Migrated {migrated} weekly folder(s) into monthly parents")

            # Step 2: Generate last month's summary if complete (async)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                result = loop.run_until_complete(generator.generate_last_month_if_complete())
                if result and result.success:
                    print(f"[MonthlyNotes] Catch-up: Generated {result.month_name} {result.year} summary ({result.daily_notes_found} daily notes)")
                elif result and result.skipped_reason:
                    print(f"[MonthlyNotes] Catch-up: Skipped ({result.skipped_reason})")
            finally:
                loop.close()

        except Exception as e:
            # Non-critical: log and continue
            print(f"[MonthlyNotes] Catch-up failed (non-critical): {e}")

    # Run in background thread so it doesn't block GUI startup
    thread = threading.Thread(target=_catchup_task, daemon=True)
    thread.start()
    print("[MonthlyNotes] Started background catch-up check...")


def _run_reference_docs_seed():
    """
    Auto-seed docs/ directory into reference_docs ChromaDB collection on startup.
    Uses content hash for idempotency — unchanged files are skipped.
    Non-blocking - runs in background daemon thread.
    """
    import threading

    def _seed_task():
        try:
            from config.app_config import (
                REFERENCE_DOCS_ENABLED, REFERENCE_DOCS_AUTO_SEED,
                REFERENCE_DOCS_SEED_PATHS,
            )
            if not REFERENCE_DOCS_ENABLED or not REFERENCE_DOCS_AUTO_SEED:
                return

            from knowledge.reference_docs_manager import ReferenceDocsManager
            from pathlib import Path

            manager = ReferenceDocsManager()
            total_uploaded = 0
            total_skipped = 0
            total_failed = 0

            for seed_path in REFERENCE_DOCS_SEED_PATHS:
                p = Path(seed_path).expanduser().resolve()
                if p.is_dir():
                    result = manager.sync_directory(str(p))
                    total_uploaded += result['uploaded']
                    total_skipped += result['skipped']
                    total_failed += result['failed']
                elif p.is_file():
                    status = manager.sync_file(str(p))
                    if status == 'uploaded':
                        total_uploaded += 1
                    elif status == 'skipped':
                        total_skipped += 1
                    else:
                        total_failed += 1
                else:
                    print(f"[RefDocs] Seed path not found: {seed_path}")

            print(
                f"[RefDocs] Auto-seed complete: "
                f"{total_uploaded} uploaded, {total_skipped} unchanged, "
                f"{total_failed} failed"
            )

        except Exception as e:
            print(f"[RefDocs] Auto-seed failed (non-critical): {e}")

    thread = threading.Thread(target=_seed_task, daemon=True)
    thread.start()
    print("[RefDocs] Started background auto-seed...")


def _launch_wizard_ui(orchestrator, share, server_name, port):
    """
    Launch the onboarding wizard UI for first-run users.

    Args:
        orchestrator: DaemonOrchestrator instance
        share: Whether to create public gradio.live link
        server_name: Server hostname/IP
        port: Server port

    Returns:
        None (launches Gradio app)
    """
    print("[DEBUG] _launch_wizard_ui called")
    print(f"[DEBUG]   share={share}, server_name={server_name}, port={port}")

    async def wizard_submit(user_input, wizard_state_dict, chat_history):
        """
        Handle wizard message submission.

        Args:
            user_input: User's message
            wizard_state_dict: Dict representation of WizardState
            chat_history: Chat history for display

        Returns:
            Tuple of (updated_chat_history, updated_wizard_state_dict, cleared_input, completion_message)
        """
        # Check if wizard was already completed (identity exists)
        # This handles page refresh after completion
        if orchestrator.user_profile and orchestrator.user_profile.identity.name:
            print("[DEBUG] Wizard already completed (identity exists), ignoring input")
            completion_msg = "✅ **Setup already complete!** Please restart the app with `python main.py` (no wizard flag) to start chatting."

            # Return current state without processing
            chat_history = list(chat_history or [])
            return chat_history, wizard_state_dict, "", completion_msg

        # Reconstruct WizardState from dict
        from gui.wizard import WizardStep
        state = WizardState(
            step=WizardStep(wizard_state_dict['step']),
            collected_data=wizard_state_dict['collected_data'],
            error_count=wizard_state_dict['error_count'],
            max_retries=wizard_state_dict['max_retries']
        )

        # Process wizard message
        response, new_state, is_complete = await process_wizard_message(user_input, state, orchestrator)

        # Update chat history
        chat_history = list(chat_history or [])
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})

        # Convert state back to dict for Gradio state
        new_state_dict = {
            'step': new_state.step.value,
            'collected_data': new_state.collected_data,
            'error_count': new_state.error_count,
            'max_retries': new_state.max_retries
        }

        # If wizard is complete, show completion message
        completion_msg = ""
        if is_complete:
            completion_msg = "✅ **Setup complete!** Close this window and relaunch the application to start chatting."

        return chat_history, new_state_dict, "", completion_msg

    print("[DEBUG] Building Gradio wizard interface...")

    # Check if wizard was already completed
    wizard_already_complete = (
        orchestrator.user_profile
        and orchestrator.user_profile.identity.name
    )

    if wizard_already_complete:
        print(f"[DEBUG] Wizard already completed (identity: '{orchestrator.user_profile.identity.name}')")
        initial_message = "✅ **Setup already complete!**\n\nYour profile has been saved. Please restart the app with `python main.py` (no wizard flag) to start chatting."
    else:
        initial_message = get_welcome_message()

    with gr.Blocks(theme=get_dark_theme(), css=DARK_CHATBOT_CSS) as demo:
        gr.Markdown("## 🤖 Daemon - First Time Setup")
        if not wizard_already_complete:
            gr.Markdown("Welcome! Let's get you set up. This should only take a minute.")
        else:
            gr.Markdown("**Note:** Wizard has already been completed. Please restart the app.")

        chatbot = gr.Chatbot(
            label="Setup Wizard",
            height=400,
            type="messages",
            value=[{"role": "assistant", "content": initial_message}]
        )

        user_input = gr.Textbox(
            lines=2,
            placeholder="Type your response here...",
            label="Your Response"
        )

        with gr.Row():
            submit_button = gr.Button("Submit", variant="primary")

        # Completion message (appears when wizard finishes)
        completion_md = gr.Markdown(value="", visible=True)

        # Wizard state stored as dict (Gradio State can't handle custom classes directly)
        # Start at WELCOME - first user input advances to INTRO, then to API_KEY
        from gui.wizard import WizardStep
        wizard_state = gr.State({
            'step': WizardStep.WELCOME.value,  # Start at WELCOME for new intro flow
            'collected_data': {},
            'error_count': 0,
            'max_retries': 3
        })

        submit_button.click(
            wizard_submit,
            inputs=[user_input, wizard_state, chatbot],
            outputs=[chatbot, wizard_state, user_input, completion_md]
        )

    # Launch wizard UI
    print("[DEBUG] Gradio interface built successfully")
    print(f"\n[*] Launching Daemon Setup Wizard on http://{server_name}:{port}")
    if share:
        print("[*] Creating public gradio.live link...")

    print("[DEBUG] Calling demo.launch()...")
    try:
        demo.launch(
            server_name=server_name,
            server_port=port,
            share=share,
            inbrowser=True,  # auto-open browser during first setup
            prevent_thread_lock=True,
        )
        print("[DEBUG] demo.launch() returned")

        # Keep alive until browser tab closes or Ctrl+C
        import threading
        _wizard_event = threading.Event()

        if IS_FROZEN:
            def _on_wizard_close():
                print("[Wizard] Browser tab closed — shutting down...")
                _wizard_event.set()
            demo.unload(_on_wizard_close)

        try:
            while not _wizard_event.is_set():
                _wizard_event.wait(timeout=1.0)
        except KeyboardInterrupt:
            print("[Wizard] Shutting down...")

    except Exception as e:
        print(f"[ERROR] demo.launch() failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def launch_gui(orchestrator, force_wizard=False):
    conversation_logger = get_conversation_logger()

    # ------- Configurable networking (via env) -------
    # GRADIO_SHARE: 1/true to request public gradio.live tunnel; 0/false for local only
    # GRADIO_SERVER_NAME: usually "127.0.0.1" (safer for local), or "0.0.0.0" for LAN
    # GRADIO_PORT: desired port (falls back to free port if occupied)
    SHARE = _env_flag("GRADIO_SHARE", True)   # default True to keep your current behavior
    SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")  # use loopback by default
    PORT = int(os.getenv("GRADIO_PORT", "7860"))
    PORT = _find_free_port(PORT)

    from config.app_config import DAEMON_MODE as _DAEMON_MODE
    _show_dev_tabs = (_DAEMON_MODE == "dev")

    # ------- First-run wizard check -------
    try:
        # force_wizard=True (from `python main.py wizard`) always forces wizard mode
        # This is useful for testing/re-running setup
        if force_wizard:
            print("[DEBUG] Force wizard mode enabled")
            is_first_run = True
        else:
            has_profile = orchestrator.user_profile is not None
            if has_profile:
                corpus_mgr = orchestrator.memory_system.corpus_manager
                is_first_run = orchestrator.user_profile.is_first_run(corpus_mgr)

                # Debug logging
                print(f"[DEBUG] First-run check:")
                print(f"  - User profile exists: {has_profile}")
                print(f"  - Corpus count: {len(corpus_mgr.corpus) if hasattr(corpus_mgr, 'corpus') else 0}")
                print(f"  - Identity name: '{orchestrator.user_profile.identity.name}'")
                print(f"  - Is first run: {is_first_run}")
            else:
                print("[DEBUG] No user profile found, skipping wizard")
                is_first_run = False
    except Exception as e:
        print(f"[DEBUG] First-run check failed: {e}")
        import traceback
        traceback.print_exc()
        is_first_run = False

    if is_first_run:
        print("[DEBUG] Launching wizard UI...")
        try:
            # Show wizard UI instead of normal chat
            return _launch_wizard_ui(orchestrator, SHARE, SERVER_NAME, PORT)
        except Exception as e:
            print(f"[ERROR] Wizard UI failed to launch: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        print("[DEBUG] Launching normal chat UI...")

    # ------- Daily notes catch-up (run in background) -------
    # Creates its own model_manager to avoid sharing httpx clients across event loops
    _run_daily_notes_catchup()

    # ------- Weekly notes catch-up (run in background) -------
    # Generates last week's summary if week is complete (today is Monday+)
    _run_weekly_notes_catchup()

    # ------- Monthly notes catch-up (run in background) -------
    # Migrates weekly folders into monthly parents, generates last month's summary
    _run_monthly_notes_catchup()

    # ------- Reference docs auto-seed (run in background) -------
    # Seeds docs/ directory into ChromaDB reference_docs collection (mtime-based idempotency)
    _run_reference_docs_seed()

    # ------- Normal chat UI (non-first-run) -------
    def get_summary_status():
        cm = orchestrator.memory_system.corpus_manager
        corpus = cm.corpus

        # Determine active cadence N from runtime consolidator (fallback 20)
        try:
            N = int(getattr(getattr(orchestrator.memory_system, 'consolidator', None), 'consolidation_threshold', 20))
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"[Launch] Could not read consolidation threshold: {e}, using default 20")
            N = 20

        # Summary/reflection identification aligned with CorpusManager logic
        def _is_reflection(e: dict) -> bool:
            typ = (e.get('type') or '').lower()
            tags = [str(t).lower() for t in (e.get('tags') or [])]
            return (typ == 'reflection') or ('type:reflection' in tags)

        def _is_summary(e: dict) -> bool:
            typ = (e.get('type') or '').lower()
            tags = [str(t).lower() for t in (e.get('tags') or [])]
            if _is_reflection(e):
                return False
            return ("summary" in typ) or ('@summary' in tags) or ('type:summary' in tags)

        total_entries = len(corpus)
        total_summaries = sum(1 for e in corpus if _is_summary(e))  # not capped at 5
        non_summary_entries = sum(1 for e in corpus if (not _is_summary(e)) and (not _is_reflection(e)))

        # Estimate backlog and remaining (t)
        def _is_consolidator_summary(e: dict) -> bool:
            if not _is_summary(e):
                return False
            tags = [str(t).lower() for t in (e.get('tags') or [])]
            return ('summary:consolidated' in tags) or ('source:consolidator' in tags)

        def _parse_block_n(tags: list):  # -> Optional[int]
            try:
                key = f"block_n:{N}:"
                for t in (tags or []):
                    t = str(t).strip().lower()
                    if t.startswith(key):
                        return int(t.split(':', 2)[2])
            except (ValueError, IndexError) as e:
                logger.debug(f"[Launch] Could not parse block_n from tags: {e}")
                return None
            return None

        tagged_blocks = set()
        for e in corpus:
            if _is_consolidator_summary(e):
                bi = _parse_block_n(e.get('tags') or [])
                if isinstance(bi, int):
                    tagged_blocks.add(bi)

        # Prefer tagged count when present; else count of consolidator summaries
        prev_blocks = len(tagged_blocks) if tagged_blocks else sum(1 for e in corpus if _is_consolidator_summary(e))
        total_blocks = non_summary_entries // N
        backlog = max(0, total_blocks - prev_blocks)

        if backlog > 0:
            since_last = 0
            remaining = 0
        else:
            since_last = non_summary_entries % N
            remaining = 0 if since_last == 0 else (N - since_last)
        conv_stats = conversation_logger.get_session_stats()
        # Reflection counts
        cm = orchestrator.memory_system.corpus_manager
        try:
            corpus_refl = [
                e for e in cm.corpus
                if (e.get("type", "").lower() == "reflection") or ("type:reflection" in (e.get("tags") or []))
            ]
        except (AttributeError, TypeError) as e:
            logger.debug(f"[Launch] Could not count corpus reflections: {e}")
            corpus_refl = []
        try:
            coll = getattr(orchestrator.memory_system, "chroma_store", None).collections.get("reflections")
            sem_refl_count = coll.count() if coll else 0
        except (AttributeError, KeyError) as e:
            logger.debug(f"[Launch] Could not count semantic reflections: {e}")
            sem_refl_count = 0
        # Last user message time (from recent corpus entries)
        def _fmt_ts(ts):
            from datetime import datetime, timezone
            try:
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if isinstance(ts, datetime):
                    if ts.tzinfo is not None and ts.tzinfo.utcoffset(ts) is not None:
                        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
                    return ts.strftime('%Y-%m-%d %H:%M')
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug(f"[Launch] Timestamp format failed: {e}")
                return ""
            return ""
        try:
            last = orchestrator.memory_system.corpus_manager.get_recent_memories(1)
            last_ts = _fmt_ts(last[0].get("timestamp")) if last else ""
        except (AttributeError, IndexError, KeyError) as e:
            logger.debug(f"[Launch] Could not get last message time: {e}")
            last_ts = ""

        return {
            "total_summaries": total_summaries,
            "total_entries": total_entries,
            "non_summary_entries": non_summary_entries,
            "summary_cadence_N": N,
            "since_last": since_last,
            "next_summary_in": remaining,
            "backlog_blocks": backlog,
            "conversation_count": conv_stats["conversation_count"],
            "log_file": conv_stats["current_log_file"],
            "reflections_corpus": len(corpus_refl),
            "reflections_semantic": sem_refl_count,
            "last_user_message_time": last_ts,
        }

    # ---- Debug Trace helpers ----
    def _format_debug_entries(entries):
        if not entries:
            return "No debug entries yet. Submit a message in Chat."

        # Pick the most recent non-empty system prompt (so it stays accurate if topic changes)
        latest_system_prompt = None
        try:
            for rec in reversed(entries):
                sp = (rec.get('system_prompt') if isinstance(rec, dict) else None) or None
                if isinstance(sp, str) and sp.strip():
                    latest_system_prompt = sp.strip()
                    break
        except (TypeError, AttributeError) as e:
            logger.debug(f"[Launch] Could not extract system prompt from debug entries: {e}")
            latest_system_prompt = None

        parts = []
        for i, rec in enumerate(entries, start=1):
            mode = rec.get('mode') or 'enhanced'
            model = rec.get('model') or ''
            q = rec.get('query') or ''
            prompt = rec.get('prompt') or ''
            resp = rec.get('response') or ''
            ptoks = rec.get('prompt_tokens')
            stoks = rec.get('system_tokens')
            ttoks = rec.get('total_tokens')
            tok_line = ""
            try:
                if ptoks is not None:
                    tok_line = f"Tokens — prompt: {ptoks}  system: {stoks or 0}  total: {ttoks or (ptoks + (stoks or 0))}"
            except (TypeError, ValueError):
                tok_line = ""
            # Build the segment for this entry
            segment = f"### #{i} — Mode: {mode}  Model: {model}\n"
            if tok_line:
                segment += f"{tok_line}\n\n"

            # Show the System Prompt exactly once at the top (below token line),
            # using the most recent value so it stays current across queries.
            # Only shown in dev mode to avoid exposing internals to users.
            if i == 1 and latest_system_prompt and _DAEMON_MODE == "dev":
                segment += f"**System Prompt**\n\n````\n{latest_system_prompt}\n````\n\n"

            segment += (
                f"**Query**\n\n````\n{q}\n````\n\n"
                f"**Prompt**\n\n````\n{prompt}\n````\n\n"
                f"**Response**\n\n````\n{resp}\n````\n\n---\n"
            )
            parts.append(segment)
        return "\n".join(parts)

    def get_recent_conversation_log(num_lines=50):
        """Read and return the last N lines of the conversation log"""
        try:
            log_path = conversation_logger.get_current_log_path()
            if log_path.exists():
                with open(log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    return ''.join(lines[-num_lines:])
            return "No conversation log yet."
        except Exception as e:
            return f"Error reading log: {e}"

    def get_app_log_path():
        """Find the app .log file from the root logger's FileHandler if present."""
        try:
            root = logging.getLogger()
            for h in root.handlers:
                if isinstance(h, logging.FileHandler):
                    # type: ignore[attr-defined]
                    return getattr(h, 'baseFilename', None)
        except (AttributeError, TypeError):
            pass
        # Fallback to the default used in utils.logging_utils.configure_logging
        return os.path.abspath('daemon_debug.log')

    def get_recent_app_log(num_lines=200):
        """Read tail of the main app .log file (daemon_debug.log by default)."""
        try:
            path = get_app_log_path()
            if not path:
                return "No file handler configured for logging."
            if not os.path.exists(path):
                return f"Log file not found at: {path}"
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
                return ''.join(lines[-num_lines:])
        except Exception as e:
            return f"Error reading app log: {e}"

    async def submit_chat(user_text, chat_history, files, use_raw_gpt, enable_citations_flag, fast_mode, personality, debug_entries):
        import logging
        logger = logging.getLogger("gradio_gui")
        logger.warning(f"[SUBMIT_CHAT] ENTRY - fast_mode={fast_mode}, type={type(fast_mode)}")

        # Update orchestrator citation mode
        orchestrator.enable_citations = enable_citations_flag

        # Ensure we have a list to work with
        chat_history = list(chat_history or [])
        # Start a new turn for type="messages": append user, then assistant placeholder
        chat_history.append({"role": "user", "content": user_text})
        chat_history.append({"role": "assistant", "content": "…"})  # ephemeral typing dots
        # Emit immediately so the user sees their message appear
        debug_entries = list(debug_entries or [])
        # Thinking state for duel mode
        thinking_a_text = ""
        thinking_b_text = ""
        thinking_visible = False
        winner_text = ""
        # Initial emit: clear input, keep debug_state only (debug view updates via state.change)
        import time as _t, asyncio as _a
        _t0 = _t.time(); _updates = 0; _last_tick = _t0
        typing_text = "<div style='text-align:right'>Assistant is typing …</div>"
        timer_text = "<div style='text-align:right'>⏱️ 0.0 s</div>"
        # Use a deep copy for the Chatbot output to avoid aliasing issues
        _chatbot_view = copy.deepcopy(chat_history)
        _state_view = copy.deepcopy(chat_history)
        yield _chatbot_view, _state_view, "", debug_entries, typing_text, timer_text, gr.update(visible=thinking_visible), thinking_a_text, thinking_b_text, winner_text

        # Concurrent loop: tick timer while awaiting streamed chunks, without blocking loop
        agen = handle_submit(
            user_text=user_text,
            files=files,
            history=chat_history,
            use_raw_gpt=use_raw_gpt,
            orchestrator=orchestrator,
            personality=personality,
            fast_mode=fast_mode
        )

        # Prime the first fetch task
        next_task = _a.create_task(agen.__anext__())
        loop_iter = 0
        while True:
            loop_iter += 1
            # Wait either for next chunk or a tick interval
            tick = _a.create_task(_a.sleep(0.25))
            done, pending = await _a.wait({next_task, tick}, return_when=_a.FIRST_COMPLETED)
            if loop_iter <= 5 or loop_iter % 20 == 0:
                logging.info(f"[GUI Loop] iter={loop_iter}, done={len(done)}, pending={len(pending)}")

            # Always handle next_task first if it's done
            if next_task in done:
                try:
                    chunk = next_task.result()
                except StopAsyncIteration:
                    break
                except Exception as e:
                    # If streaming errored, log it and show error to user
                    logging.error(f"[GUI] Streaming error: {type(e).__name__}: {e}")
                    import traceback
                    logging.error(f"[GUI] Traceback:\n{traceback.format_exc()}")

                    error_msg = f"⚠️ Connection error: {str(e)}"
                    if chat_history and isinstance(chat_history[-1], dict):
                        chat_history[-1]["content"] = error_msg

                    typing_text = ""
                    timer_text = f"<div style='text-align:right'>⏱️ {_t.time() - _t0:.1f} s</div>"
                    _chatbot_view = copy.deepcopy(chat_history)
                    _state_view = copy.deepcopy(chat_history)
                    yield _chatbot_view, _state_view, "", debug_entries, typing_text, timer_text, gr.update(visible=thinking_visible), thinking_a_text, thinking_b_text, winner_text
                    break

                # Process streamed chunk
                if isinstance(chunk, dict) and "thinking" in chunk:
                    # Handle thinking data from duel mode - embed in chat message
                    logging.info(f"[GUI] THINKING CHUNK RECEIVED: {chunk.get('thinking', {}).keys()}")
                    think = chunk["thinking"]
                    model_a = think.get('model_a', 'Model A')
                    model_b = think.get('model_b', 'Model B')
                    thinking_a = think.get('thinking_a', '')
                    thinking_b = think.get('thinking_b', '')
                    winner = think.get('winner', '')
                    scores = think.get('scores', {})

                    # Build embedded thinking display as HTML details/summary
                    thinking_html = "<details open><summary>💭 <b>Thinking Process (Duel Mode)</b></summary>\n\n"
                    thinking_html += f"**{model_a}:**\n{thinking_a}\n\n"
                    thinking_html += f"**{model_b}:**\n{thinking_b}\n\n"
                    if winner and scores:
                        thinking_html += f"🏆 **Winner: Model {winner}** (Scores: A={scores.get('A', 'N/A')}, B={scores.get('B', 'N/A')})\n"
                    thinking_html += "</details>\n\n"

                    # Update chat with thinking content
                    if chat_history and isinstance(chat_history[-1], dict):
                        chat_history[-1]["content"] = thinking_html

                    # Yield immediately to show thinking (use deep copies to avoid aliasing)
                    _chatbot_view = copy.deepcopy(chat_history)
                    _state_view = copy.deepcopy(chat_history)
                    yield _chatbot_view, _state_view, "", debug_entries, typing_text, timer_text, gr.update(visible=False), "", "", ""
                elif isinstance(chunk, dict) and "content" in chunk:
                    assistant_reply = chunk["content"]
                    logging.debug(f"[GUI LAUNCH] Processing chunk: '{assistant_reply[:100] if assistant_reply else 'EMPTY'}'")
                    # Update the last assistant message's content
                    if assistant_reply:  # Only update if there's actual content
                        if chat_history and isinstance(chat_history[-1], dict):
                            # If there's already thinking HTML, preserve it and add answer after
                            current_content = chat_history[-1].get("content", "")
                            if current_content and "<details" in current_content:
                                # Extract thinking block and append new answer
                                if "</details>" in current_content:
                                    # Split at the end of details tag
                                    parts = current_content.split("</details>\n\n", 1)
                                    thinking_part = parts[0] + "</details>\n\n"
                                    chat_history[-1]["content"] = thinking_part + assistant_reply
                                else:
                                    chat_history[-1]["content"] = current_content + assistant_reply
                            else:
                                chat_history[-1]["content"] = assistant_reply
                        else:
                            # Fallback in case state shape is unexpected
                            chat_history.append({"role": "assistant", "content": assistant_reply})
                else:
                    assistant_reply = str(chunk)
                    if chat_history and isinstance(chat_history[-1], dict):
                        chat_history[-1]["content"] = assistant_reply
                    else:
                        chat_history.append({"role": "assistant", "content": assistant_reply})

                if isinstance(chunk, dict) and "debug" in chunk:
                    try:
                        debug_entries.append(chunk["debug"])  # append single record
                    except (TypeError, KeyError):
                        pass
                    # Fallback: if no assistant content was streamed, set it from debug.response
                    try:
                        if (not isinstance(chunk, dict)) or ("content" not in chunk or not chunk.get("content")):
                            final_from_debug = (chunk.get("debug") or {}).get("response")
                            if final_from_debug:
                                if chat_history and isinstance(chat_history[-1], dict) and chat_history[-1].get("role") == "assistant":
                                    chat_history[-1]["content"] = final_from_debug
                    except (TypeError, KeyError, IndexError):
                        pass
                # Re-yield current state along with typing + timer
                now = _t.time(); _updates += 1
                if (now - _last_tick) >= 0.10 or (_updates % 2 == 0):
                    _last_tick = now
                    _dots = "." * (1 + (_updates % 3))
                    typing_text = f"<div style='text-align:right'>Assistant is typing {_dots}</div>"
                    timer_text = f"<div style='text-align:right'>⏱️ {now - _t0:.1f} s</div>"
                    _chatbot_view = copy.deepcopy(chat_history)
                    _state_view = copy.deepcopy(chat_history)
                    yield _chatbot_view, _state_view, "", debug_entries, typing_text, timer_text, gr.update(visible=thinking_visible), thinking_a_text, thinking_b_text, winner_text

                # Schedule next chunk read
                next_task = _a.create_task(agen.__anext__())
            elif tick in done:
                # Timer tick: update indicators only if next_task wasn't processed
                now = _t.time(); _updates += 1
                if (now - _last_tick) >= 0.20 or (_updates % 3 == 0):
                    _last_tick = now
                    _dots = "." * (1 + (_updates % 3))
                    typing_text = f"<div style='text-align:right'>Assistant is typing {_dots}</div>"
                    timer_text = f"<div style='text-align:right'>⏱️ {now - _t0:.1f} s</div>"
                    _chatbot_view = copy.deepcopy(chat_history)
                    _state_view = copy.deepcopy(chat_history)
                    yield _chatbot_view, _state_view, "", debug_entries, typing_text, timer_text, gr.update(visible=thinking_visible), thinking_a_text, thinking_b_text, winner_text

        # Final update: clear typing indicator, freeze timer
        typing_text = ""
        timer_text = f"<div style='text-align:right'>⏱️ {_t.time() - _t0:.1f} s</div>"
        _chatbot_view = copy.deepcopy(chat_history)
        _state_view = copy.deepcopy(chat_history)
        yield _chatbot_view, _state_view, "", debug_entries, typing_text, timer_text, gr.update(visible=thinking_visible), thinking_a_text, thinking_b_text, winner_text

    # ---- Settings persistence helpers ----
    def _load_settings():
        """Load persisted UI settings from config/config.yaml (best-effort)."""
        try:
            import yaml  # type: ignore
            from pathlib import Path
            cfg_path = Path('config') / 'config.yaml'
            if cfg_path.exists():
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
        except (IOError, OSError, ImportError):
            pass
        return {}

    def _save_settings(updater):
        """Update config/config.yaml with a callable that mutates the dict."""
        try:
            import yaml  # type: ignore
            from pathlib import Path
            cfg_path = Path('config') / 'config.yaml'
            data = {}
            if cfg_path.exists():
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
            updater(data)
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cfg_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(data, f, sort_keys=False)
            return True, None
        except Exception as e:
            return False, str(e)

    # Apply persisted active model at startup (if present)
    try:
        _persisted = (_load_settings().get('models', {}) or {}).get('active')
        if isinstance(_persisted, str) and _persisted.strip():
            try:
                orchestrator.model_manager.switch_model(_persisted.strip())
            except (AttributeError, ValueError):
                pass
    except (AttributeError, TypeError, KeyError):
        pass

    # Apply persisted default temperature at startup (if present)
    try:
        _m = (_load_settings().get('models', {}) or {})
        _t = _m.get('default_temperature', None)
        if _t is not None:
            try:
                orchestrator.model_manager.default_temperature = float(_t)
            except (ValueError, TypeError, AttributeError):
                pass
    except (AttributeError, TypeError, KeyError):
        pass

    with gr.Blocks(theme=get_dark_theme(), css=DARK_CHATBOT_CSS) as demo:
        gr.Markdown("## 🤖 Daemon Chat Interface")

        with gr.Tabs():
            with gr.TabItem("Chat"):
                chatbot = gr.Chatbot(label="Daemon", height=520, type="messages")
                # Thinking process display for duel mode (two models side-by-side)
                with gr.Accordion("💭 Thinking Process", open=True, visible=False) as thinking_accordion:
                    gr.Markdown("### Model Reasoning Comparison")
                    with gr.Row():
                        with gr.Column():
                            model_a_label = gr.Markdown("**Model A**")
                            thinking_a_md = gr.Markdown(value="", elem_id="thinking_a")
                        with gr.Column():
                            model_b_label = gr.Markdown("**Model B**")
                            thinking_b_md = gr.Markdown(value="", elem_id="thinking_b")
                    winner_md = gr.Markdown(value="", elem_id="winner_display")
                # Place typing + timer directly under the chat so they stay visible
                typing_md = gr.Markdown(value="", elem_id="typing_indicator")
                timer_md = gr.Markdown(value="", elem_id="response_timer")
                user_input = gr.Textbox(lines=2, placeholder="Ask Daemon something...", label="Your Message")
                with gr.Row():
                    submit_button = gr.Button("Submit", variant="primary")
                    clear_button = gr.Button("🧹 Clear Chat")
                    sync_notes_button = gr.Button("📝 Sync Notes")

                    # --- Active Model selector (on chat page for visibility) ---
                    try:
                        _mm = orchestrator.model_manager
                        _api_aliases = list(getattr(_mm, 'api_models', {}).keys())
                        _local_models = list(getattr(_mm, 'models', {}).keys())
                        _model_choices = sorted(set(_api_aliases + _local_models)) or [_mm.get_active_model_name() or 'gpt-4-turbo']
                        _current_active = _mm.get_active_model_name() or (_model_choices[0] if _model_choices else 'gpt-4-turbo')
                    except (AttributeError, TypeError, KeyError):
                        _model_choices = ['gpt-5.1', 'gpt-5', 'gpt-4-turbo', 'claude-opus-4.5', 'claude-opus', 'sonnet-4.6', 'sonnet-4.5']
                        _current_active = 'gpt-5.1'

                    model_dd = gr.Dropdown(
                        label="Model",
                        choices=_model_choices,
                        value=_current_active,
                        interactive=True,
                        scale=1,
                    )

                sync_status_md = gr.Markdown(value="", elem_id="sync_status")
                model_status = gr.Markdown(visible=True)

                with gr.Row():
                    files = gr.File(file_types=[".txt", ".docx", ".csv", ".py", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp"], file_count="multiple", label="Files & Images")
                    use_raw = gr.Checkbox(label="Bypass Memory (Raw GPT)", value=False)
                    enable_citations = gr.Checkbox(label="Enable Memory Citations", value=False, info="Show which memories Claude references")
                    fast_mode = gr.Checkbox(label="⚡ Fast Mode", value=False, info="Reduced context for mobile/slow connections (~2x faster)")
                    personality = gr.State("default")

                chat_state = gr.State([])
                debug_state = gr.State([])

                # Immediately preset typing + timer on click so they appear before streaming starts
                def _preset_typing():
                    return (
                        "<div style='text-align:right'>Assistant is typing …</div>",
                        "<div style='text-align:right'>⏱️ 0.0 s</div>",
                        gr.update(visible=False),
                        "",
                        "",
                        ""
                    )

                submit_button.click(
                    _preset_typing,
                    inputs=[],
                    outputs=[typing_md, timer_md, thinking_accordion, thinking_a_md, thinking_b_md, winner_md],
                )

                submit_button.click(
                    submit_chat,
                    inputs=[user_input, chat_state, files, use_raw, enable_citations, fast_mode, personality, debug_state],
                    outputs=[chatbot, chat_state, user_input, debug_state, typing_md, timer_md, thinking_accordion, thinking_a_md, thinking_b_md, winner_md],
                )

                # Clear chat handler
                def _clear_chat():
                    return [], [], "", "", "", gr.update(visible=False), "", "", ""

                clear_button.click(
                    fn=_clear_chat,
                    inputs=[],
                    outputs=[chatbot, chat_state, user_input, typing_md, timer_md, thinking_accordion, thinking_a_md, thinking_b_md, winner_md],
                )

                # Sync Obsidian notes handler
                def _sync_obsidian_notes():
                    try:
                        from knowledge.obsidian_manager import ObsidianManager
                        manager = ObsidianManager()
                        result = manager.embed_vault(force_reindex=False)

                        if result.errors:
                            return f"⚠️ Sync completed with errors: {', '.join(result.errors)}"
                        elif result.embedded_files == 0 and result.updated_files == 0 and result.skipped_files > 0:
                            return f"✓ All {result.skipped_files} notes unchanged"
                        else:
                            parts = []
                            if result.embedded_files:
                                parts.append(f"{result.embedded_files} new")
                            if result.updated_files:
                                parts.append(f"{result.updated_files} updated")
                            return f"✅ Synced {', '.join(parts)} notes ({result.total_chunks} chunks) in {result.duration_seconds:.1f}s. Skipped {result.skipped_files} unchanged."
                    except Exception as e:
                        return f"❌ Sync failed: {str(e)}"

                sync_notes_button.click(
                    fn=_sync_obsidian_notes,
                    inputs=[],
                    outputs=[sync_status_md],
                )

                # Model selector change handler (fires on dropdown change)
                def _apply_active_model(name: str):
                    _name = (name or '').strip()
                    if not _name:
                        return "No model selected."
                    try:
                        _mm = orchestrator.model_manager
                        _mm.switch_model(_name)
                        def _update_active(d):
                            if 'models' not in d:
                                d['models'] = {}
                            d['models']['active'] = _name
                        _ok, _err = _save_settings(_update_active)
                        if not _ok:
                            return f"Switched to '{_name}'. Persist failed: {_err}"
                        return f"Model: {_name}"
                    except Exception as _e:
                        return f"Failed to switch model: {_e}"

                model_dd.change(_apply_active_model, inputs=[model_dd], outputs=[model_status])

            with gr.TabItem("Logs", visible=_show_dev_tabs):
                gr.Markdown("### 📜 Live Logs")
                with gr.Row():
                    log_source = gr.Dropdown(
                        label="Source",
                        choices=["App Log (.log)", "Conversation Log"],
                        value="App Log (.log)",
                    )
                    auto_refresh = gr.Checkbox(label="Auto‑refresh", value=True)
                tail_n = gr.Slider(label="Tail lines", minimum=50, maximum=5000, step=50, value=400)

                # Helpful path hints
                app_path_md = gr.Markdown(value=f"App log: `{get_app_log_path()}`", visible=True)
                conv_path_md = gr.Markdown(value=f"Conversation log: `{conversation_logger.get_current_log_path()}`", visible=False)

                def _path_hints(src):
                    # Toggle visibility: show only the relevant path row
                    return (
                        gr.update(visible=(src == "App Log (.log)")),
                        gr.update(visible=(src == "Conversation Log")),
                    )

                log_source.change(_path_hints, inputs=[log_source], outputs=[app_path_md, conv_path_md])

                # Log viewer: use None for plain text (no highlighting)
                log_view = gr.Code(value=get_recent_app_log(400), language=None, label="Tail")

                def _read_logs(src: str, n: int):
                    n = int(n or 200)
                    if src == "Conversation Log":
                        return get_recent_conversation_log(n)
                    return get_recent_app_log(n)

                refresh_btn = gr.Button("🔄 Refresh")
                refresh_btn.click(_read_logs, inputs=[log_source, tail_n], outputs=[log_view])

                # Auto refresh using a timer; toggle via checkbox
                # Gradio >=5 uses `value` for the interval (seconds); older versions used `interval`.
                try:
                    timer = gr.Timer(value=2.0, active=True)
                except TypeError:
                    timer = gr.Timer(interval=2.0, active=True)
                auto_refresh.change(lambda v: gr.update(active=bool(v)), inputs=[auto_refresh], outputs=[timer])
                timer.tick(_read_logs, inputs=[log_source, tail_n], outputs=[log_view])
            with gr.TabItem("Debug Trace", visible=_show_dev_tabs):
                gr.Markdown("### 🔎 Query → Prompt → Response")
                debug_view = gr.Markdown(value=_format_debug_entries([]))

                # Download button for full prompt
                with gr.Row():
                    download_prompt_btn = gr.Button("📥 Download Full Prompt as TXT", variant="secondary")
                    download_status = gr.Markdown(value="", visible=True)
                download_file = gr.File(label="Download", visible=False)

                def _download_full_prompt(entries):
                    """Extract the most recent full prompt and prepare it for download"""
                    try:
                        if not entries:
                            return gr.update(visible=False), "❌ No debug entries available. Submit a message first."

                        # Get the most recent entry
                        latest = entries[-1] if isinstance(entries, list) else entries

                        # Extract components
                        system_prompt = latest.get('system_prompt', '')
                        prompt = latest.get('prompt', '')
                        query = latest.get('query', '')
                        mode = latest.get('mode', 'unknown')
                        model = latest.get('model', 'unknown')

                        # Build the full prompt text
                        lines = []
                        lines.append("="*80)
                        lines.append("DAEMON RAG AGENT - FULL PROMPT EXPORT")
                        lines.append("="*80)
                        lines.append(f"Mode: {mode}")
                        lines.append(f"Model: {model}")
                        lines.append("="*80)
                        lines.append("")

                        if system_prompt and _DAEMON_MODE == "dev":
                            lines.append("[SYSTEM PROMPT]")
                            lines.append("-"*80)
                            lines.append(system_prompt)
                            lines.append("")
                            lines.append("="*80)
                            lines.append("")

                        lines.append("[USER QUERY]")
                        lines.append("-"*80)
                        lines.append(query)
                        lines.append("")
                        lines.append("="*80)
                        lines.append("")

                        lines.append("[FULL CONTEXT PROMPT]")
                        lines.append("-"*80)
                        lines.append(prompt)
                        lines.append("")
                        lines.append("="*80)

                        content = "\n".join(lines)

                        # Write to temporary file
                        from pathlib import Path
                        import tempfile
                        from datetime import datetime

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"daemon_prompt_{timestamp}.txt"

                        # Use a temporary directory
                        temp_dir = Path(tempfile.gettempdir()) / "daemon_prompts"
                        temp_dir.mkdir(exist_ok=True)
                        filepath = temp_dir / filename

                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)

                        return gr.update(value=str(filepath), visible=True), f"✅ Prompt exported to {filename}"

                    except Exception as e:
                        import traceback
                        logging.error(f"[GUI] Download prompt error: {e}\n{traceback.format_exc()}")
                        return gr.update(visible=False), f"❌ Error: {str(e)}"

                download_prompt_btn.click(
                    _download_full_prompt,
                    inputs=[debug_state],
                    outputs=[download_file, download_status]
                )

                # Update debug view whenever debug_state changes
                def _render_debug(entries):
                    return _format_debug_entries(entries)
                # Bind state change to view update
                debug_state.change(fn=_render_debug, inputs=[debug_state], outputs=[debug_view])

            with gr.TabItem("Provenance"):
                gr.Markdown("### Provenance")
                provenance_view = gr.JSON(value={}, label="Provenance")

                def _format_provenance(entries):
                    """Extract and format provenance from debug entries"""
                    if not entries:
                        return {}
                    latest = entries[-1] if isinstance(entries, list) else entries
                    if not isinstance(latest, dict):
                        return {}
                    prov = dict(latest.get('provenance', {})) if latest.get('provenance') else {}
                    # Always include basic fields from debug_record
                    prov['mode'] = latest.get('mode', '')
                    prov['model'] = latest.get('model', '')
                    prov['citations_enabled'] = latest.get('citations_enabled', False)
                    citations = latest.get('citations', [])
                    if citations:
                        prov['citations'] = citations
                    if latest.get('prompt_tokens'):
                        prov['prompt_tokens'] = latest['prompt_tokens']
                    if latest.get('total_tokens'):
                        prov['total_tokens'] = latest['total_tokens']
                    # Truncate thinking_block for display (full version in ChromaDB)
                    _GUI_THINKING_CAP = 500
                    for _tb_key in ('thinking_block', 'thinking_a', 'thinking_b'):
                        if _tb_key in prov and isinstance(prov[_tb_key], str) and len(prov[_tb_key]) > _GUI_THINKING_CAP:
                            prov[_tb_key] = prov[_tb_key][:_GUI_THINKING_CAP] + " [truncated]"
                    return prov

                # Update provenance view whenever debug_state changes
                debug_state.change(fn=_format_provenance, inputs=[debug_state], outputs=[provenance_view])

            with gr.TabItem("Status"):
                gr.Markdown("### 📊 Runtime Status")
                with gr.Row():
                    summary_json = gr.JSON(value=get_summary_status(), label="Status")
                    refresh_button = gr.Button("🔄 Refresh Status")
                refresh_button.click(fn=get_summary_status, outputs=summary_json)

                # --- Memory Maintenance (dev mode only — user mode auto-dedup on shutdown) ---
                if _show_dev_tabs:
                    gr.Markdown("---")
                    gr.Markdown("### Memory Maintenance")
                    with gr.Row():
                        dedup_preview_btn = gr.Button("Preview Dedup", variant="secondary")
                        dedup_execute_btn = gr.Button("Run Dedup", variant="primary")
                    dedup_report_md = gr.Markdown(value="*Click Preview to scan for duplicates.*")

                    def _run_dedup(dry_run: bool) -> str:
                        try:
                            from memory.cross_deduplicator import CrossCollectionDeduplicator
                            store = orchestrator.memory_system.chroma_store
                            dedup = CrossCollectionDeduplicator(store)
                            plan = dedup.run(dry_run=dry_run)
                            return plan.to_markdown()
                        except Exception as e:
                            return f"**Error:** {e}"

                    dedup_preview_btn.click(
                        fn=lambda: _run_dedup(dry_run=True),
                        outputs=dedup_report_md,
                    )
                    dedup_execute_btn.click(
                        fn=lambda: _run_dedup(dry_run=False),
                        outputs=dedup_report_md,
                    )

            # Proposals tab (extracted to gui/tabs/proposals.py)
            from gui.tabs.proposals import build_proposals_tab
            _proposals = build_proposals_tab(orchestrator, _load_settings, _save_settings, _show_dev_tabs)

            # Synthesis tab (extracted to gui/tabs/synthesis.py)
            from gui.tabs.synthesis import build_synthesis_tab
            build_synthesis_tab(orchestrator, _show_dev_tabs)

            # Settings tab (extracted to gui/tabs/settings.py)
            from gui.tabs.settings import build_settings_tab
            build_settings_tab(orchestrator, _load_settings, _save_settings)

            # Original inline Proposals/Synthesis/Settings tab code removed.
            # Now in gui/tabs/proposals.py, gui/tabs/synthesis.py, gui/tabs/settings.py

            # ===============================================================
            # TAB: Personality
            # ===============================================================
            with gr.TabItem("Personality"):
                gr.Markdown("### Personality Instructions")
                gr.Markdown(
                    "Edit the personality section of Daemon's system prompt. "
                    "Operating principles (facts handling, memory integration, "
                    "guardrails, etc.) are appended automatically and cannot be edited here."
                )

                def _load_current_personality():
                    try:
                        from config.app_config import load_personality_text
                        return load_personality_text()
                    except Exception:
                        return ""

                personality_textbox = gr.Textbox(
                    label="Personality Prompt",
                    value=_load_current_personality(),
                    lines=25,
                    max_lines=50,
                    interactive=True,
                )

                with gr.Row():
                    set_personality_btn = gr.Button("Set", variant="primary")
                    restore_personality_btn = gr.Button("Restore Default", variant="secondary")

                personality_status_md = gr.Markdown(value="")

                def _set_personality(text):
                    try:
                        from config.app_config import PERSONALITY_CUSTOM_PATH, PERSONALITY_MAX_CHARS
                        from pathlib import Path
                        if len(text) > PERSONALITY_MAX_CHARS:
                            logger.warning(f"[Personality] Rejected save: {len(text)} chars exceeds limit of {PERSONALITY_MAX_CHARS}")
                            return f"Too long ({len(text)} chars). Max is {PERSONALITY_MAX_CHARS}. Trim and retry."
                        Path(PERSONALITY_CUSTOM_PATH).parent.mkdir(parents=True, exist_ok=True)
                        with open(PERSONALITY_CUSTOM_PATH, "w", encoding="utf-8") as f:
                            f.write(text)
                        logger.info(f"[Personality] Custom personality saved ({len(text)} chars) to {PERSONALITY_CUSTOM_PATH}")
                        return f"Custom personality saved ({len(text)} chars). Takes effect on the next message."
                    except Exception as e:
                        logger.error(f"[Personality] Failed to save custom personality: {e}")
                        return f"Failed to save: {e}"

                def _restore_default_personality():
                    try:
                        from config.app_config import PERSONALITY_CUSTOM_PATH, load_default_personality
                        from pathlib import Path
                        custom = Path(PERSONALITY_CUSTOM_PATH)
                        if custom.exists():
                            custom.unlink()
                            logger.info(f"[Personality] Deleted custom personality file: {PERSONALITY_CUSTOM_PATH}")
                        default_text = load_default_personality()
                        logger.info(f"[Personality] Restored default personality ({len(default_text)} chars)")
                        return default_text, "Restored default personality. Takes effect on the next message."
                    except Exception as e:
                        logger.error(f"[Personality] Failed to restore default: {e}")
                        return gr.update(), f"Failed to restore: {e}"

                set_personality_btn.click(
                    _set_personality,
                    inputs=[personality_textbox],
                    outputs=[personality_status_md],
                )
                restore_personality_btn.click(
                    _restore_default_personality,
                    inputs=[],
                    outputs=[personality_textbox, personality_status_md],
                )

        # --- Auto-load proposals on page open ---
        demo.load(
            _proposals["load_proposals"],
            inputs=[_proposals["status_filter"], _proposals["type_filter"]],
            outputs=[_proposals["proposals_view"], _proposals["manage_selector"],
                     _proposals["codegen_selector"], _proposals["proposals_map"]],
        )

    # --- Configure queue with timeout ---
    # Configure for mobile compatibility: api_open keeps SSE alive during slow prepare_prompt
    demo.queue(
        default_concurrency_limit=10,
        max_size=120,
        api_open=True  # Keep API connections open during long context building (mobile fix)
    )

    # --- Launch logic with graceful fallback ---
    # prevent_thread_lock so we can inspect URLs or handle fallback
    try:
        app, local_url, share_url = demo.launch(
            server_name=SERVER_NAME,
            server_port=PORT,
            max_file_size="100mb",
            max_threads=40,
            quiet=False,
            share=SHARE,
            inbrowser=_env_flag("GRADIO_OPEN_BROWSER", IS_FROZEN),  # Auto-open for desktop app
            prevent_thread_lock=True,
        )
        # Add health check endpoint
        from utils.health_check import add_health_endpoint
        add_health_endpoint(app, orchestrator)

        # Print URLs for visibility
        logger = logging.getLogger("gui.launch")
        print(f"[GUI] Local:  {local_url}")
        logger.warning(f"[GUI] Local:  {local_url}")
        print(f"[GUI] Health: {local_url}/health")
        logger.warning(f"[GUI] Health: {local_url}/health")
        if SHARE and share_url:
            print(f"[GUI] Public: {share_url}")
            logger.warning(f"[GUI] Public: {share_url}")
            print(f"[GUI] Health: {share_url}/health")
            logger.warning(f"[GUI] Share Health: {share_url}/health")
        elif SHARE and not share_url:
            print("[GUI] Requested share=True but no share URL returned.")
            logger.warning("[GUI] Requested share=True but no share URL returned.")

        # Robust browser opening for frozen executable (handles icon launch)
        # Uses platform-specific commands which are more reliable than webbrowser module
        if IS_FROZEN and _env_flag("GRADIO_OPEN_BROWSER", True):
            import subprocess
            import time
            time.sleep(0.5)  # Brief delay to ensure server is ready
            try:
                if sys.platform.startswith('linux'):
                    subprocess.Popen(['xdg-open', local_url],
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', local_url])
                elif sys.platform == 'win32':
                    os.startfile(local_url)
                print(f"[GUI] Browser opened: {local_url}")
            except Exception as e:
                print(f"[GUI] Could not auto-open browser: {e}")
                print(f"[GUI] Please open manually: {local_url}")

    except Exception as e:
        # If public tunnel requested but failed (common with SSL/proxy), fall back to local
        logger = logging.getLogger("gui.launch")
        print(f"[GUI] Launch error: {e}")
        logger.error(f"[GUI] Launch error: {e}")
        if SHARE:
            print("[GUI] Falling back to local (share=False). "
                  "Tip: set GRADIO_SHARE=0 to skip public tunnel.")
            logger.warning("[GUI] Falling back to local (share=False). "
                  "Tip: set GRADIO_SHARE=0 to skip public tunnel.")
            app, local_url, _ = demo.launch(
                server_name=SERVER_NAME,
                server_port=PORT,
                max_file_size="100mb",
                max_threads=40,
                quiet=False,
                share=False,
                inbrowser=_env_flag("GRADIO_OPEN_BROWSER", IS_FROZEN),  # Auto-open for desktop app
                prevent_thread_lock=True,
            )
            # Add health check endpoint (fallback path)
            from utils.health_check import add_health_endpoint
            add_health_endpoint(app, orchestrator)

            print(f"[GUI] Local: {local_url}")
            print(f"[GUI] Health: {local_url}/health")

            # Robust browser opening for frozen executable (fallback path)
            if IS_FROZEN and _env_flag("GRADIO_OPEN_BROWSER", True):
                import subprocess
                import time
                time.sleep(0.5)
                try:
                    if sys.platform.startswith('linux'):
                        subprocess.Popen(['xdg-open', local_url],
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
                    elif sys.platform == 'darwin':
                        subprocess.Popen(['open', local_url])
                    elif sys.platform == 'win32':
                        os.startfile(local_url)
                    print(f"[GUI] Browser opened: {local_url}")
                except Exception as browser_err:
                    print(f"[GUI] Could not auto-open browser: {browser_err}")
                    print(f"[GUI] Please open manually: {local_url}")
        else:
            raise

    # When running as desktop app, shut down when browser tab closes
    import threading
    shutdown_event = threading.Event()

    if IS_FROZEN:
        def _on_browser_close():
            print("[GUI] Browser tab closed — shutting down...")
            shutdown_event.set()

        demo.unload(_on_browser_close)

    # Keep main thread alive with responsive shutdown handling
    try:
        while not shutdown_event.is_set():
            shutdown_event.wait(timeout=1.0)
    except KeyboardInterrupt:
        print("[GUI] Shutting down...")
