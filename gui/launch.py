"""
# gui/launch.py

Module Contract
- Purpose: Define and launch the Gradio UI. Provides Chat, Debug Trace, Status, and Settings tabs; wires UI events to handlers and state.
- Inputs:
  - Orchestrator instance (built in main). Environment flags for networking.
- Outputs:
  - Running web app (local or shared); exposes live chat and debug views.
- Key pieces:
  - submit_chat(): async driver that streams tokens + timer updates
  - Tabs:
    • Chat: chat UI, file upload, raw toggle, personality selector
    • Debug Trace: renders debug_state entries
    • Status: shows counters (summaries, corpus counts, logs)
    • Settings: slider to control summary cadence (every N exchanges)
- Side effects:
  - None beyond UI + logging.
"""
import os
import logging
import socket
import gradio as gr
import copy
from gui.handlers import handle_submit
from utils.conversation_logger import get_conversation_logger

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

def launch_gui(orchestrator):
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

    async def submit_chat(user_text, chat_history, files, use_raw_gpt, personality, debug_entries):
        personality_manager.switch_personality(personality)

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

            if tick in done:
                # Timer tick: update indicators
                now = _t.time(); _updates += 1
                if (now - _last_tick) >= 0.20 or (_updates % 3 == 0):
                    _last_tick = now
                    _dots = "." * (1 + (_updates % 3))
                    typing_text = f"<div style='text-align:right'>Assistant is typing {_dots}</div>"
                    timer_text = f"<div style='text-align:right'>⏱️ {now - _t0:.1f} s</div>"
                    _chatbot_view = copy.deepcopy(chat_history)
                    _state_view = copy.deepcopy(chat_history)
                    yield _chatbot_view, _state_view, "", debug_entries, typing_text, timer_text, gr.update(visible=thinking_visible), thinking_a_text, thinking_b_text, winner_text
                # Continue; next_task may also be done
            if next_task in done:
                try:
                    chunk = next_task.result()
                except StopAsyncIteration:
                    break
                except Exception:
                    # If streaming errored, stop typing and break
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

                    # Yield immediately to show thinking
                    yield chat_history, chat_history, "", debug_entries, typing_text, timer_text, gr.update(visible=False), "", "", ""
                elif isinstance(chunk, dict) and "content" in chunk:
                    assistant_reply = chunk["content"]
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
                # Re-yield current state along with typing + timer
                now = _t.time(); _updates += 1
                if (now - _last_tick) >= 0.10 or (_updates % 2 == 0):
                    _last_tick = now
                    _dots = "." * (1 + (_updates % 3))
                    typing_text = f"<div style='text-align:right'>Assistant is typing {_dots}</div>"
                    timer_text = f"<div style='text-align:right'>⏱️ {now - _t0:.1f} s</div>"
                    yield chat_history, chat_history, "", debug_entries, typing_text, timer_text, gr.update(visible=thinking_visible), thinking_a_text, thinking_b_text, winner_text

                # Schedule next chunk read
                next_task = _a.create_task(agen.__anext__())

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
                    inputs=[user_input, chat_state, files, use_raw, personality, debug_state],
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

                # Update debug view whenever debug_state changes
                def _render_debug(entries):
                    return _format_debug_entries(entries)
                # Bind state change to view update
                debug_state.change(fn=_render_debug, inputs=[debug_state], outputs=[debug_view])

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
                    _model_choices = ['gpt-5', 'gpt-4-turbo', 'claude-opus']
                    _current_active = 'gpt-5'

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
                        _ok, _err = _save_settings(lambda d: d.setdefault('models', {}).update({'active': _name}))
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
                    _all_model_choices = ['gpt-5', 'gpt-4-turbo', 'claude-opus', 'gpt-4o', 'gpt-4o-mini']

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
            inbrowser=_env_flag("GRADIO_OPEN_BROWSER", False),
            prevent_thread_lock=True,
        )
        # Print URLs for visibility
        print(f"[GUI] Local:  {local_url}")
        if SHARE and share_url:
            print(f"[GUI] Public: {share_url}")
        elif SHARE and not share_url:
            print("[GUI] Requested share=True but no share URL returned.")

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
                inbrowser=_env_flag("GRADIO_OPEN_BROWSER", False),
                prevent_thread_lock=True,
            )
            print(f"[GUI] Local: {local_url}")
        else:
            raise

    # Keep main thread alive if needed
    try:
        import time
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("[GUI] Shutting down...")
