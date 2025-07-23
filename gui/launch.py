import gradio as gr
from gui.handlers import handle_submit

def launch_gui(orchestrator):
    personality_manager = orchestrator.personality_manager

    def get_summary_status():
        summaries = orchestrator.memory_system.corpus_manager.get_summaries()
        total_entries = len(orchestrator.memory_system.corpus_manager.corpus)
        non_summary_entries = len([
            e for e in orchestrator.memory_system.corpus_manager.corpus
            if "@summary" not in e.get("tags", [])
        ])
        return {
            "total_summaries": len(summaries),
            "total_entries": total_entries,
            "non_summary_entries": non_summary_entries,
            "next_summary_in": 20 - (non_summary_entries % 20)
        }

    async def submit_chat(user_text, chat_history, files, use_raw_gpt, personality):
        personality_manager.switch_personality(personality)
        async for chunk in handle_submit(
            user_text=user_text,
            files=files,
            history=chat_history,
            use_raw_gpt=use_raw_gpt,
            orchestrator=orchestrator
        ):
            # If chunk is a dict, extract 'content'
            if isinstance(chunk, dict) and "content" in chunk:
                assistant_reply = chunk["content"]
            else:
                assistant_reply = str(chunk)
            yield chat_history + [[user_text, assistant_reply]]


    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("## 🤖 Daemon Chat Interface")
        with gr.Row():
            summary_json = gr.JSON(value=get_summary_status(), label="📊 Summary Status")
            refresh_button = gr.Button("🔄 Refresh Summary Status")
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

    demo.launch(
        server_name="0.0.0.0",
        max_file_size="100mb",
        max_threads=40,
        quiet=False,
        share=True
    )
