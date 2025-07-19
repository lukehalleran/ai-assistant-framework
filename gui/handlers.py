import pandas as pd
import logging
from utils.logging_utils import log_and_time
import json
from config.config import SYSTEM_PROMPT as system_prompt
logger = logging.getLogger("gradio_gui")


def smart_join(prev: str, new: str) -> str:
    """
    Inserts a space between tokens unless the new chunk begins with punctuation or whitespace.
    Prevents jammed-together words while respecting formatting.
    """
    if not prev:
        return new
    if prev.endswith((' ', '\n')) or new.startswith((' ', '\n', '.', ',', '?', '!', "'", '"', ")", "’", "”")):
        return prev + new
    else:
        return prev + ' ' + new


@log_and_time("Handle Submit")
async def handle_submit(user_text, files, history, use_raw_gpt, orchestrator):
    logger.debug(f"[Handle Submit] Received user_text: {user_text}")


    if not user_text.strip():
        yield {"role": "assistant", "content": "⚠️ Empty input received."}
        return

    # Process files
    file_data = []
    if files:
        for file in files:
            try:
                if file.name.endswith(".txt"):
                    with open(file.name, 'r') as f:
                        file_data.append(f.read())
                elif file.name.endswith(".csv"):
                    df = pd.read_csv(file.name)
                    file_data.append(df.to_string())
                elif file.name.endswith(".py"):
                    with open(file.name, 'r') as f:
                        file_data.append(f.read())
                else:
                    file_data.append(f"[Unsupported file type: {file.name}]")
            except Exception as e:
                file_data.append(f"[Error reading {file.name}: {str(e)}]")

    # Step 1: Merge file content into user_input
    user_input = user_text + "\n\n" + "\n\n".join(file_data)

    # Step 2: Build full hierarchical prompt
    full_prompt = await orchestrator.prompt_builder.build_hierarchical_prompt(
        user_input=user_input,
        include_dreams=True,
        include_wiki=True,
        include_semantic=True,
    )

    logger.debug(f"[Handle Submit] Final prompt being passed to model:\n{full_prompt}")

    # Step 3: Stream response from full prompt
    try:
        logger.debug(f"[🔍 FINAL MESSAGE PAYLOAD TO OPENAI]:\n{json.dumps([{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': full_prompt}], indent=2)}")

        final_output = ""
        async for chunk in orchestrator.response_generator.generate_streaming_response(
            prompt=full_prompt,
            model_name=orchestrator.model_manager.get_active_model_name()
        ):

            final_output = smart_join(final_output, chunk)
            yield {"role": "assistant", "content": final_output}
                # ✅ This ensures Gradio gets *something* in the end
        if not final_output:
            yield {"role": "assistant", "content": "⚠️ No response was generated."}

    except Exception as e:
        logger.error(f"[HANDLE_SUBMIT] Streaming error: {e}")
        yield {"role": "assistant", "content": f"⚠️ Streaming error: {str(e)}"}
