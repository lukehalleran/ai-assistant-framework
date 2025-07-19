# gui/launch.py
import gradio as gr
from gui.handlers import handle_submit


def launch_gui(orchestrator):
    personality_manager = orchestrator.personality_manager

    async def chat_interface_fn(message, history, files=None, use_raw_gpt=False, personality="default"):
        # Switch personality first
        personality_manager.switch_personality(personality)
        # Run generator-based streaming
        async for chunk in handle_submit(
            user_text=message,
            files=files,
            history=history,
            use_raw_gpt=use_raw_gpt,
            orchestrator=orchestrator
        ):
            yield chunk

    # Define ChatInterface with side panel for settings
    demo = gr.ChatInterface(
        fn=chat_interface_fn,
        chatbot=gr.Chatbot(elem_id="chatbot", type="messages"),
        textbox=gr.Textbox(placeholder="Ask Daemon something...", lines=2),
        title="🤖 Daemon Chat Interface",
        description="Real-time assistant with memory and personality support",
        additional_inputs=[
            gr.File(file_types=[".txt", ".docx", ".csv", ".py"], file_count="multiple", label="Files"),
            gr.Checkbox(label="Bypass Memory (Raw GPT)", value=False),
            gr.Dropdown(
                label="Personality",
                choices=list(personality_manager.personalities.keys()),
                value=personality_manager.current_personality
            )
        ],
        theme="soft",
        fill_height=True,
        examples=[
            ["What's the capital of France?"],
            ["Summarize the plot of *Frankenstein*."],
            ["Explain how cosine similarity is used in memory gating."]
        ]
    )

    demo.launch(
        server_name="0.0.0.0",
        max_file_size="100mb",
        max_threads=40,
        quiet=False,
        share=True
    )
