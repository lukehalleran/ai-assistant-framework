"""
# gui/launch.py

Module Contract
- Purpose: Define and launch the Gradio UI. Provides Chat, Debug Trace, Status, and Settings tabs;
  wires UI events to handlers and state. Registers health check endpoint. Routes to wizard for first-run.
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
  - submit_chat(): async driver that streams tokens + timer updates
  - Tabs:
    • Chat: chat UI, file upload, raw toggle, personality selector
    • Debug Trace: renders debug_state entries
    • Status: shows counters (summaries, corpus counts, logs)
    • Settings: slider to control summary cadence (every N exchanges)
  - Health endpoint: Lightweight checks (corpus, ChromaDB, API key) for Docker/K8s
- Dependencies:
  - gui.handlers (handle_submit)
  - gui.wizard (WizardState, process_wizard_message)
  - utils.health_check (add_health_endpoint)
  - utils.conversation_logger
- Side effects:
  - Registers /health endpoint on FastAPI app after launch
  - Logs URLs and health endpoint to console
  - [FROZEN MODE] Auto-opens browser via platform-specific command (xdg-open/open/startfile)
  - Runs indefinite event loop (until KeyboardInterrupt)
- Threading/Async: Main thread blocked by event loop; Gradio handles async internally
"""
import os
import sys
import logging
import socket
import gradio as gr
import copy
from gui.handlers import handle_submit
from utils.conversation_logger import get_conversation_logger
from gui.wizard import WizardState, process_wizard_message, get_welcome_message

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
            completion_msg = "✅ **Setup complete!** Please restart the app with `python main.py` (no wizard flag) to start chatting."

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

    with gr.Blocks(theme="soft") as demo:
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
        # Start at API_KEY since welcome message is already displayed in initial chatbot value
        from gui.wizard import WizardStep
        wizard_state = gr.State({
            'step': WizardStep.API_KEY.value,  # Start at API_KEY, not WELCOME
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
            inbrowser=True  #  auto-open browser during first setup
        )
        print("[DEBUG] demo.launch() returned")
    except Exception as e:
        print(f"[ERROR] demo.launch() failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def launch_gui(orchestrator, force_wizard=False):
    personality_manager = orchestrator.personality_manager
    conversation_logger = get_conversation_logger()

    # ------- Configurable networking (via env) -------
    # GRADIO_SHARE: 1/true to request public gradio.live tunnel; 0/false for local only
    # GRADIO_SERVER_NAME: usually "127.0.0.1" (safer for local), or "0.0.0.0" for LAN
    # GRADIO_PORT: desired port (falls back to free port if occupied)
    SHARE = _env_flag("GRADIO_SHARE", True)   # default True to keep your current behavior
    SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")  # use loopback by default
    PORT = int(os.getenv("GRADIO_PORT", "7860"))
    PORT = _find_free_port(PORT)

    # ------- First-run wizard check -------
    try:
        # If wizard was completed (identity.name exists), ignore force_wizard flag
        # This prevents wizard from re-running on refresh after completion
        if force_wizard:
            has_profile = orchestrator.user_profile is not None
            if has_profile and orchestrator.user_profile.identity.name:
                print("[DEBUG] Force wizard mode enabled BUT wizard already completed (identity exists)")
                print(f"[DEBUG]   - Identity name: '{orchestrator.user_profile.identity.name}'")
                print("[DEBUG]   - Ignoring force_wizard flag, will launch normal chat")
                force_wizard = False  # Override - wizard already completed

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

    # ------- Normal chat UI (non-first-run) -------
    def get_summary_status():
        cm = orchestrator.memory_system.corpus_manager
        corpus = cm.corpus

        # Determine active cadence N from runtime consolidator (fallback 20)
        try:
            N = int(getattr(getattr(orchestrator.memory_system, 'consolidator', None), 'consolidation_threshold', 20))
        except Exception:
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
            except Exception:
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
        except Exception:
            corpus_refl = []
        try:
            coll = getattr(orchestrator.memory_system, "chroma_store", None).collections.get("reflections")
            sem_refl_count = coll.count() if coll else 0
        except Exception:
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
            except Exception:
                return ""
            return ""
        try:
            last = orchestrator.memory_system.corpus_manager.get_recent_memories(1)
            last_ts = _fmt_ts(last[0].get("timestamp")) if last else ""
        except Exception:
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
        except Exception:
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
            except Exception:
                tok_line = ""
            # Build the segment for this entry
            segment = f"### #{i} — Mode: {mode}  Model: {model}\n"
            if tok_line:
                segment += f"{tok_line}\n\n"

            # Show the System Prompt exactly once at the top (below token line),
            # using the most recent value so it stays current across queries.
            if i == 1 and latest_system_prompt:
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
        except Exception:
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

    async def submit_chat(user_text, chat_history, files, use_raw_gpt, enable_citations_flag, personality, debug_entries):
        personality_manager.switch_personality(personality)

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
            personality=personality
        )

        # Prime the first fetch task
        next_task = _a.create_task(agen.__anext__())
        while True:
            # Wait either for next chunk or a tick interval
            tick = _a.create_task(_a.sleep(0.25))
            done, pending = await _a.wait({next_task, tick}, return_when=_a.FIRST_COMPLETED)

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
                    logging.warning(f"[GUI LAUNCH DEBUG] Processing chunk content: '{assistant_reply[:100] if assistant_reply else 'EMPTY'}'")
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
                    except Exception:
                        pass
                    # Fallback: if no assistant content was streamed, set it from debug.response
                    try:
                        if (not isinstance(chunk, dict)) or ("content" not in chunk or not chunk.get("content")):
                            final_from_debug = (chunk.get("debug") or {}).get("response")
                            if final_from_debug:
                                if chat_history and isinstance(chat_history[-1], dict) and chat_history[-1].get("role") == "assistant":
                                    chat_history[-1]["content"] = final_from_debug
                    except Exception:
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
        except Exception:
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
            except Exception:
                pass
    except Exception:
        pass

    # Apply persisted default temperature at startup (if present)
    try:
        _m = (_load_settings().get('models', {}) or {})
        _t = _m.get('default_temperature', None)
        if _t is not None:
            try:
                orchestrator.model_manager.default_temperature = float(_t)
            except Exception:
                pass
    except Exception:
        pass

    with gr.Blocks(theme="soft") as demo:
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

                with gr.Row():
                    files = gr.File(file_types=[".txt", ".docx", ".csv", ".py"], file_count="multiple", label="Files")
                    use_raw = gr.Checkbox(label="Bypass Memory (Raw GPT)", value=False)
                    enable_citations = gr.Checkbox(label="Enable Memory Citations", value=False, info="Show which memories Claude references")
                    personality = gr.Dropdown(
                        label="Personality",
                        choices=list(personality_manager.personalities.keys()),
                        value=personality_manager.current_personality
                    )

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
                    inputs=[user_input, chat_state, files, use_raw, enable_citations, personality, debug_state],
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

            with gr.TabItem("Logs"):
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
            with gr.TabItem("Debug Trace"):
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

                        if system_prompt:
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

            with gr.TabItem("Citations"):
                gr.Markdown("### 📚 Memory Citations")
                gr.Markdown("Shows which memories Claude referenced in responses (when citation mode enabled)")
                citations_view = gr.JSON(value=[], label="Citations")

                def _format_citations(entries):
                    """Extract and format citations from debug entries"""
                    if not entries:
                        return []
                    # Get latest entry's citations
                    latest = entries[-1] if isinstance(entries, list) else entries
                    return latest.get('citations', [])

                # Update citations view whenever debug_state changes
                debug_state.change(fn=_format_citations, inputs=[debug_state], outputs=[citations_view])

            with gr.TabItem("Status"):
                gr.Markdown("### 📊 Runtime Status")
                with gr.Row():
                    summary_json = gr.JSON(value=get_summary_status(), label="Status")
                    refresh_button = gr.Button("🔄 Refresh Status")
                refresh_button.click(fn=get_summary_status, outputs=summary_json)

            with gr.TabItem("Settings"):
                gr.Markdown("### ⚙️ Runtime Settings")

                # --- Active Model selector (persisted) ---
                try:
                    _mm = orchestrator.model_manager
                    _api_aliases = list(getattr(_mm, 'api_models', {}).keys())
                    _local_models = list(getattr(_mm, 'models', {}).keys())
                    _model_choices = sorted(set(_api_aliases + _local_models)) or [_mm.get_active_model_name() or 'gpt-4-turbo']
                    _current_active = _mm.get_active_model_name() or (_model_choices[0] if _model_choices else 'gpt-4-turbo')
                except Exception:
                    _model_choices = ['gpt-5.1', 'gpt-5', 'gpt-4-turbo', 'claude-opus-4.5', 'claude-opus', 'sonnet-4.5']
                    _current_active = 'gpt-5.1'

                with gr.Row():
                    model_dd = gr.Dropdown(
                        label="Active Model",
                        choices=_model_choices,
                        value=_current_active,
                        interactive=True,
                    )
                apply_model_btn = gr.Button("Set Active Model", variant="primary")
                model_status = gr.Markdown(visible=True)
                gr.Markdown("Note: Active Model is used for normal streaming and as a fallback. Duel mode runs two models in parallel and lets a judge pick the winner.")

                def _apply_active_model(name: str):
                    _name = (name or '').strip()
                    if not _name:
                        return "No model selected."
                    try:
                        _mm = orchestrator.model_manager
                        _mm.switch_model(_name)
                        # Fixed: Make sure the update actually modifies the dict
                        def _update_active(d):
                            if 'models' not in d:
                                d['models'] = {}
                            d['models']['active'] = _name
                        _ok, _err = _save_settings(_update_active)
                        if not _ok:
                            return f"Switched to '{_name}'. Persist failed: {_err}"
                        return f"Active model set to: {_name} (persisted)."
                    except Exception as _e:
                        return f"Failed to switch model: {_e}"

                apply_model_btn.click(_apply_active_model, inputs=[model_dd], outputs=[model_status])

                # --- Faster Streaming options ---
                gr.Markdown("### ⚡ Faster Streaming (reduce first-token latency)")
                # Load defaults from config for initial checkbox states
                try:
                    _settings = _load_settings()
                    _feat = (_settings.get('features', {}) or {})
                    _disable_bestof_default = not bool(_feat.get('enable_best_of', True))
                    _disable_rewrite_default = not bool(_feat.get('enable_query_rewrite', True))
                    _disable_summaries_default = bool(_feat.get('disable_llm_summaries', False))
                except Exception:
                    _disable_bestof_default = False
                    _disable_rewrite_default = False
                    _disable_summaries_default = False

                with gr.Row():
                    disable_bestof = gr.Checkbox(label="Disable Best-of (no multi-sample reranking)", value=_disable_bestof_default)
                    disable_rewrite = gr.Checkbox(label="Disable Query Rewrite (skip pre-call)", value=_disable_rewrite_default)
                    disable_summaries = gr.Checkbox(label="Disable LLM Summaries (skip pre-call)", value=_disable_summaries_default)
                try:
                    _bo_budget_default = float((_settings.get('features', {}) or {}).get('best_of_latency_budget_s', 0))
                except Exception:
                    _bo_budget_default = 0.0
                bestof_budget = gr.Slider(label="Best‑of Latency Budget (s)", minimum=0.0, maximum=120.0, step=0.5, value=_bo_budget_default)
                apply_fast_btn = gr.Button("Apply Streaming Settings")
                fast_status = gr.Markdown(visible=True)

                def _apply_fast(disable_bo: bool, disable_rw: bool, disable_sums: bool, bo_budget: float):
                    try:
                        # Update runtime config on the orchestrator
                        cfg = getattr(orchestrator, 'config', {}) or {}
                        feats = cfg.setdefault('features', {}) if isinstance(cfg, dict) else {}
                        feats['enable_best_of'] = not bool(disable_bo)
                        feats['enable_query_rewrite'] = not bool(disable_rw)
                        feats['disable_llm_summaries'] = bool(disable_sums)
                        feats['best_of_latency_budget_s'] = float(bo_budget)
                        # Attempt runtime override for prompt builder (no restart needed)
                        try:
                            pb = getattr(orchestrator, 'prompt_builder', None)
                            if pb is not None:
                                setattr(pb, 'force_llm_summaries', False if disable_sums else True)
                        except Exception:
                            pass
                        # Persist to YAML
                        ok, err = _save_settings(lambda d: d.setdefault('features', {}).update({
                            'enable_best_of': not bool(disable_bo),
                            'enable_query_rewrite': not bool(disable_rw),
                            'disable_llm_summaries': bool(disable_sums),
                            'best_of_latency_budget_s': float(bo_budget),
                        }))
                        if not ok:
                            return f"Applied runtime settings. Persist failed: {err}"
                        return "Streaming settings updated (persisted)."
                    except Exception as e:
                        return f"Failed to apply: {e}"

                apply_fast_btn.click(_apply_fast, inputs=[disable_bestof, disable_rewrite, disable_summaries, bestof_budget], outputs=[fast_status])

                # --- Web Search Settings ---
                gr.Markdown("### 🔍 Web Search (Tavily API)")
                try:
                    _ws_settings = _load_settings()
                    _ws_cfg = (_ws_settings.get('web_search', {}) or {})
                    _ws_enabled_default = bool(_ws_cfg.get('enabled', True))
                    _ws_daily_limit_default = int(_ws_cfg.get('daily_credit_limit', 100))
                except Exception:
                    _ws_enabled_default = True
                    _ws_daily_limit_default = 100

                with gr.Row():
                    web_search_enabled = gr.Checkbox(
                        label="Enable Web Search",
                        value=_ws_enabled_default,
                        info="Search the web for queries requiring current information"
                    )
                    web_search_daily_limit = gr.Slider(
                        label="Daily Credit Limit",
                        minimum=10,
                        maximum=500,
                        step=10,
                        value=_ws_daily_limit_default,
                        info="Maximum credits to use per day (Tavily free tier: ~33/day)"
                    )

                apply_ws_btn = gr.Button("Apply Web Search Settings")
                ws_status = gr.Markdown(visible=True)

                def _apply_web_search(enabled: bool, daily_limit: int):
                    try:
                        # Update runtime config
                        cfg = getattr(orchestrator, 'config', {}) or {}
                        ws_cfg = cfg.setdefault('web_search', {}) if isinstance(cfg, dict) else {}
                        ws_cfg['enabled'] = bool(enabled)
                        ws_cfg['daily_credit_limit'] = int(daily_limit)

                        # Update global config if available
                        try:
                            import config.app_config as app_cfg
                            app_cfg.WEB_SEARCH_ENABLED = bool(enabled)
                            app_cfg.WEB_SEARCH_DAILY_CREDIT_LIMIT = int(daily_limit)
                        except Exception:
                            pass

                        # Persist to YAML
                        ok, err = _save_settings(lambda d: d.setdefault('web_search', {}).update({
                            'enabled': bool(enabled),
                            'daily_credit_limit': int(daily_limit),
                        }))
                        if not ok:
                            return f"Applied runtime settings. Persist failed: {err}"
                        return f"Web search settings updated: enabled={enabled}, daily_limit={daily_limit}"
                    except Exception as e:
                        return f"Failed to apply: {e}"

                apply_ws_btn.click(
                    _apply_web_search,
                    inputs=[web_search_enabled, web_search_daily_limit],
                    outputs=[ws_status]
                )

                # --- Best-of / Duel Mode (runtime + persisted) ---
                gr.Markdown("### 🥊 Best‑of / Duel Mode")
                try:
                    _settings = _load_settings()
                    _feat = (_settings.get('features', {}) or {})
                    _duel_enabled_default = bool(_feat.get('best_of_duel_mode', False))
                    _gens_default = list(_feat.get('best_of_generator_models', []))
                except Exception:
                    _duel_enabled_default = False
                    _gens_default = []

                # Build model choices from known API aliases + local models
                try:
                    _mm = orchestrator.model_manager
                    _api_aliases = list(getattr(_mm, 'api_models', {}).keys())
                    _local_models = list(getattr(_mm, 'models', {}).keys())
                    _all_model_choices = sorted(set(_api_aliases + _local_models)) or [_mm.get_active_model_name() or 'gpt-4-turbo']
                except Exception:
                    _all_model_choices = ['gpt-5.1', 'gpt-5', 'gpt-4-turbo', 'claude-opus-4.5', 'claude-opus', 'sonnet-4.5', 'gpt-4o', 'gpt-4o-mini']

                _m1_value = _gens_default[0] if len(_gens_default) > 0 else (_all_model_choices[0] if _all_model_choices else None)
                # Pick a second default different from the first, if possible
                _m2_value = _gens_default[1] if len(_gens_default) > 1 else (next((m for m in _all_model_choices if m != _m1_value), _m1_value))

                with gr.Row():
                    duel_enable = gr.Checkbox(label="Enable Duel Mode (two models + judge)", value=_duel_enabled_default)
                with gr.Row():
                    duel_model_1 = gr.Dropdown(label="Model 1", choices=_all_model_choices, value=_m1_value, interactive=True)
                    duel_model_2 = gr.Dropdown(label="Model 2", choices=_all_model_choices, value=_m2_value, interactive=True)
                apply_duel_btn = gr.Button("Apply Duel Settings", variant="primary")
                duel_status = gr.Markdown(visible=True)

                def _apply_duel_settings(enable: bool, m1: str, m2: str):
                    try:
                        m1 = (m1 or '').strip()
                        m2 = (m2 or '').strip()
                        if enable and (not m1 or not m2):
                            return "Select both Model 1 and Model 2."
                        if enable and m1 == m2:
                            return "Pick two different models for duel mode."

                        # Runtime update
                        cfg = getattr(orchestrator, 'config', {}) or {}
                        feats = cfg.setdefault('features', {}) if isinstance(cfg, dict) else {}
                        feats['best_of_duel_mode'] = bool(enable)
                        feats['best_of_generator_models'] = [m1, m2] if m1 and m2 else feats.get('best_of_generator_models', [])

                        # Persist to YAML
                        def _updater(d):
                            f = d.setdefault('features', {})
                            f['best_of_duel_mode'] = bool(enable)
                            if m1 and m2:
                                f['best_of_generator_models'] = [m1, m2]
                        ok, err = _save_settings(_updater)
                        if not ok:
                            return f"Applied runtime duel settings. Persist failed: {err}"
                        return f"Duel mode={'ON' if enable else 'OFF'} | Model 1={m1 or '-'} Model 2={m2 or '-'} (persisted)."
                    except Exception as e:
                        return f"Failed to apply: {e}"

                apply_duel_btn.click(_apply_duel_settings, inputs=[duel_enable, duel_model_1, duel_model_2], outputs=[duel_status])

                # --- Max Tokens Controls (runtime + persisted) ---
                gr.Markdown("### ✂️ Max Tokens (length/speed)")
                try:
                    _settings = _load_settings()
                    _feat = (_settings.get('features', {}) or {})
                    _models_cfg = (_settings.get('models', {}) or {})
                    _gen_maxtok_default = int(_feat.get('best_of_max_tokens', 128))
                    _judge_maxtok_default = int(_feat.get('best_of_selector_max_tokens', 64))
                    _stream_maxtok_default = int(_models_cfg.get('default_max_tokens', getattr(orchestrator.model_manager, 'default_max_tokens', 2048)))
                except Exception:
                    _gen_maxtok_default = 128
                    _judge_maxtok_default = 64
                    _stream_maxtok_default = getattr(orchestrator.model_manager, 'default_max_tokens', 2048)

                with gr.Row():
                    gen_max_tok = gr.Slider(label="Duel/Best‑of Max Tokens (per answer)", minimum=16, maximum=10000, step=16, value=_gen_maxtok_default)
                    judge_max_tok = gr.Slider(label="Judge Max Tokens", minimum=16, maximum=2000, step=8, value=_judge_maxtok_default)
                # Streaming max tokens as a separate, clear control
                stream_max_tok = gr.Slider(label="Streaming Max Tokens (final answer)", minimum=256, maximum=10000, step=64, value=_stream_maxtok_default)
                apply_tok_btn = gr.Button("Apply Token Settings", variant="primary")
                tok_status = gr.Markdown(visible=True)

                def _apply_tokens(gen_max: int, judge_max: int, stream_max: int):
                    try:
                        gen_max = int(gen_max); judge_max = int(judge_max); stream_max = int(stream_max)
                        # Runtime update
                        cfg = getattr(orchestrator, 'config', {}) or {}
                        feats = cfg.setdefault('features', {}) if isinstance(cfg, dict) else {}
                        models_cfg = cfg.setdefault('models', {}) if isinstance(cfg, dict) else {}
                        feats['best_of_max_tokens'] = gen_max
                        feats['best_of_selector_max_tokens'] = judge_max
                        models_cfg['default_max_tokens'] = stream_max
                        # Apply runtime to ModelManager
                        try:
                            mm = getattr(orchestrator, 'model_manager', None)
                            if mm is not None:
                                setattr(mm, 'default_max_tokens', int(stream_max))
                        except Exception:
                            pass
                        # Persist to YAML
                        def _updater(d):
                            f = d.setdefault('features', {})
                            f['best_of_max_tokens'] = int(gen_max)
                            f['best_of_selector_max_tokens'] = int(judge_max)
                            m = d.setdefault('models', {})
                            m['default_max_tokens'] = int(stream_max)
                        ok, err = _save_settings(_updater)
                        if not ok:
                            return f"Applied runtime tokens. Persist failed: {err}"
                        return f"Applied: generators={gen_max} judge={judge_max} streaming={stream_max} (persisted)."
                    except Exception as e:
                        return f"Failed to apply: {e}"

                apply_tok_btn.click(_apply_tokens, inputs=[gen_max_tok, judge_max_tok, stream_max_tok], outputs=[tok_status])

                # --- Model Temperature (runtime + persisted) ---
                gr.Markdown("### 🌡️ Model Temperature")
                try:
                    _settings = _load_settings()
                    _models = (_settings.get('models', {}) or {})
                    _temp_default = float(_models.get('default_temperature', getattr(orchestrator.model_manager, 'default_temperature', 0.7)))
                except Exception:
                    _temp_default = getattr(orchestrator.model_manager, 'default_temperature', 0.7)

                with gr.Row():
                    model_temp = gr.Slider(
                        label="Model Temperature",
                        minimum=0.0,
                        maximum=1.5,
                        step=0.05,
                        value=_temp_default,
                    )
                    apply_temp_btn = gr.Button("Apply Temperature", variant="primary")
                temp_status = gr.Markdown(visible=True)

                def _apply_temperature(t: float):
                    try:
                        t = float(t)
                        # Apply runtime
                        try:
                            mm = getattr(orchestrator, 'model_manager', None)
                            if mm is not None:
                                setattr(mm, 'default_temperature', t)
                        except Exception:
                            pass
                        # Persist to YAML
                        ok, err = _save_settings(lambda d: d.setdefault('models', {}).update({'default_temperature': float(t)}))
                        if not ok:
                            return f"Applied runtime temperature={t:.2f}. Persist failed: {err}"
                        return f"Model temperature set to {t:.2f} (persisted)."
                    except Exception as e:
                        return f"Failed to apply: {e}"

                apply_temp_btn.click(_apply_temperature, inputs=[model_temp], outputs=[temp_status])

                # Summary cadence (every N exchanges)
                try:
                    current_n = int(getattr(getattr(orchestrator, 'memory_system', None), 'consolidator', None).consolidation_threshold)
                except Exception:
                    current_n = 10

                with gr.Row():
                    summary_n = gr.Slider(
                        label="Summary Every N Exchanges (applies at shutdown)",
                        minimum=1,
                        maximum=200,
                        step=1,
                        value=current_n,
                    )
                    apply_btn = gr.Button("Apply", variant="primary")
                status_md = gr.Markdown(visible=True)

                def _apply_summary_n(n: int):
                    import os
                    from pathlib import Path
                    import yaml
                    try:
                        n = int(n)
                        # Update memory coordinator consolidator (used at shutdown)
                        try:
                            mc = getattr(orchestrator, 'memory_system', None)
                            if mc and getattr(mc, 'consolidator', None):
                                mc.consolidator.consolidation_threshold = n
                        except Exception:
                            pass

                        # Update prompt builder consolidator for consistency
                        try:
                            pb = getattr(orchestrator, 'prompt_builder', None)
                            if pb and getattr(pb, 'consolidator', None):
                                pb.consolidator.consolidation_threshold = n
                        except Exception:
                            pass

                        # Keep an env hint (session-scoped)
                        try:
                            os.environ['SUMMARY_EVERY_N'] = str(n)
                        except Exception:
                            pass

                        # Persist to config/config.yaml under memory.summary_interval
                        try:
                            cfg_path = Path('config') / 'config.yaml'
                            data = {}
                            if cfg_path.exists():
                                with open(cfg_path, 'r', encoding='utf-8') as f:
                                    data = yaml.safe_load(f) or {}
                            mem = data.setdefault('memory', {})
                            mem['summary_interval'] = int(n)
                            with open(cfg_path, 'w', encoding='utf-8') as f:
                                yaml.safe_dump(data, f, sort_keys=False)
                        except Exception as _e:
                            return f"Applied (runtime). Persist failed: {_e}"

                        return f"Summary cadence updated: every {n} exchanges (persisted)."
                    except Exception as e:
                        return f"Failed to apply: {e}"

                apply_btn.click(_apply_summary_n, inputs=[summary_n], outputs=[status_md])

    # --- Configure queue with timeout ---
    # Set max_size to allow long-running streaming responses (120s timeout)
    demo.queue(default_concurrency_limit=10, max_size=120)

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
        print(f"[GUI] Local:  {local_url}")
        print(f"[GUI] Health: {local_url}/health")
        if SHARE and share_url:
            print(f"[GUI] Public: {share_url}")
            print(f"[GUI] Health: {share_url}/health")
        elif SHARE and not share_url:
            print("[GUI] Requested share=True but no share URL returned.")

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
        print(f"[GUI] Launch error: {e}")
        if SHARE:
            print("[GUI] Falling back to local (share=False). "
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

    # Keep main thread alive if needed
    try:
        import time
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("[GUI] Shutting down...")
