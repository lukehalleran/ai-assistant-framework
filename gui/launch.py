# /gui/launch.py
import os
import socket
import gradio as gr
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
        summaries = orchestrator.memory_system.corpus_manager.get_summaries()
        total_entries = len(orchestrator.memory_system.corpus_manager.corpus)
        non_summary_entries = len([
            e for e in orchestrator.memory_system.corpus_manager.corpus
            if "@summary" not in e.get("tags", [])
        ])
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
            "total_summaries": len(summaries),
            "total_entries": total_entries,
            "non_summary_entries": non_summary_entries,
            "next_summary_in": 20 - (non_summary_entries % 20),
            "conversation_count": conv_stats["conversation_count"],
            "log_file": conv_stats["current_log_file"],
            "reflections_corpus": len(corpus_refl),
            "reflections_semantic": sem_refl_count,
            "last_user_message_time": last_ts,
        }

    def get_recent_reflections(n=5):
        """Return a small list of recent reflections from corpus and semantic store."""
        out = {"corpus": [], "semantic": []}
        try:
            cm = orchestrator.memory_system.corpus_manager
            items = cm.get_items_by_type("reflection", limit=n)
            out["corpus"] = [i.get("content", "") for i in items]
        except Exception:
            pass
        try:
            store = getattr(orchestrator.memory_system, "chroma_store", None)
            coll = store.collections.get("reflections") if store else None
            if coll and coll.count() > 0:
                items = store.get_recent("reflections", limit=n)
                out["semantic"] = [(i.get("content") or "") for i in items]
        except Exception:
            pass
        return out

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

    async def submit_chat(user_text, chat_history, files, use_raw_gpt, personality):
        personality_manager.switch_personality(personality)

        async for chunk in handle_submit(
            user_text=user_text,
            files=files,
            history=chat_history,
            use_raw_gpt=use_raw_gpt,
            orchestrator=orchestrator,
            personality=personality
        ):
            if isinstance(chunk, dict) and "content" in chunk:
                assistant_reply = chunk["content"]
            else:
                assistant_reply = str(chunk)
            yield chat_history + [[user_text, assistant_reply]]

    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("## 🤖 Daemon Chat Interface")

        with gr.Tabs():
            with gr.TabItem("Chat"):
                with gr.Row():
                    summary_json = gr.JSON(value=get_summary_status(), label="📊 Status")
                    refresh_button = gr.Button("🔄 Refresh Status")
                refresh_button.click(fn=get_summary_status, outputs=summary_json)

                chatbot = gr.Chatbot(label="Daemon")
                user_input = gr.Textbox(lines=2, placeholder="Ask Daemon something...", label="Your Message")
                submit_button = gr.Button("Submit")

                with gr.Row():
                    files = gr.File(file_types=[".txt", ".docx", ".csv", ".py"], file_count="multiple", label="Files")
                    use_raw = gr.Checkbox(label="Bypass Memory (Raw GPT)", value=False)
                    personality = gr.Dropdown(
                        label="Personality",
                        choices=list(personality_manager.personalities.keys()),
                        value=personality_manager.current_personality
                    )

                chat_state = gr.State([])

                submit_button.click(
                    submit_chat,
                    inputs=[user_input, chat_state, files, use_raw, personality],
                    outputs=[chatbot],
                )

            with gr.TabItem("Reflections"):
                gr.Markdown("### 🪞 Recent Reflections")
                ref_json = gr.JSON(value=get_recent_reflections(), label="Reflections (corpus & semantic)")
                with gr.Row():
                    ref_n = gr.Slider(1, 20, value=5, step=1, label="How many")
                    ref_refresh = gr.Button("🔄 Refresh Reflections")
                def _update_reflections(k):
                    return get_recent_reflections(int(k))
                ref_refresh.click(fn=_update_reflections, inputs=[ref_n], outputs=[ref_json])

            with gr.TabItem("Conversation Log"):
                gr.Markdown("### 📝 Recent Conversation History")
                gr.Markdown(f"Log file: `{conversation_logger.get_current_log_path()}`")

                with gr.Row():
                    lines_slider = gr.Slider(10, 200, value=50, step=10, label="Lines to display")
                    refresh_log_button = gr.Button("🔄 Refresh Log")

                conversation_log_display = gr.Textbox(
                    value=get_recent_conversation_log(),
                    label="Conversation Log",
                    lines=25,
                    max_lines=30,
                    interactive=False
                )

                def update_log(num_lines):
                    return get_recent_conversation_log(num_lines)

                refresh_log_button.click(
                    fn=update_log,
                    inputs=[lines_slider],
                    outputs=[conversation_log_display]
                )

                # Auto-refresh every 5 seconds when on this tab
                conversation_log_display.change(
                    fn=lambda: get_recent_conversation_log(lines_slider.value),
                    inputs=[],
                    outputs=[conversation_log_display],
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
