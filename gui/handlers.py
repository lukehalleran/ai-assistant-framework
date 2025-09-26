"""
# gui/handlers.py

Module Contract
- Purpose: Orchestrates a single chat submission in the GUI: preprocesses files, routes to raw/enhanced flows, streams the response to the UI, and persists interaction + debug trace.
- Inputs:
  - handle_submit(user_text, files, history, use_raw_gpt, orchestrator, system_prompt=?, force_summarize=?, include_summaries=?, personality=?)
- Outputs:
  - Yields streaming dicts {role, content, debug?} as Gradio updates.
- Behavior:
  - RAW mode: send directly through orchestrator.process_user_query(use_raw_mode=True)
  - ENHANCED: orchestrator.prepare_prompt → response_generator.generate_streaming_response → store interaction
- Side effects:
  - Writes to conversation logger; stores to memory_system; updates debug_state for Debug Trace tab.
"""
import pandas as pd
import logging
from utils.logging_utils import log_and_time
from utils.conversation_logger import get_conversation_logger
import json
from config.app_config import load_system_prompt, ENABLE_BEST_OF, BEST_OF_N, BEST_OF_TEMPS, BEST_OF_MAX_TOKENS, BEST_OF_MIN_QUESTION, BEST_OF_MODEL, BEST_OF_MIN_TOKENS
from utils.query_checker import analyze_query
DEFAULT_SYSTEM_PROMPT = load_system_prompt()
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
async def handle_submit(
    user_text,
    files,
    history,
    use_raw_gpt,
    orchestrator,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    force_summarize=False,
    include_summaries=True,
    personality=None
):
    logger.debug(f"[Handle Submit] Received user_text: {user_text}")

    # Get conversation logger
    conversation_logger = get_conversation_logger()

    if not user_text.strip():
        yield {"role": "assistant", "content": "⚠️ Empty input received."}
        return

    # Process files into the 'file_data' list
    file_data = []
    file_names = []  # Track file names for metadata
    if files:
        for file in files:
            file_names.append(file.name)
            try:
                if file.name.endswith(".txt"):
                    with open(file.name, 'r', encoding='utf-8') as f:
                        file_data.append(f.read())
                elif file.name.endswith(".csv"):
                    df = pd.read_csv(file.name)
                    file_data.append(df.to_string())
                elif file.name.endswith(".py"):
                    with open(file.name, 'r', encoding='utf-8') as f:
                        file_data.append(f.read())
                else:
                    file_data.append(f"[Unsupported file type: {file.name}]")
            except Exception as e:
                file_data.append(f"[Error reading {file.name}: {str(e)}]")

    # Merge file content into user_input
    merged_input = user_text + "\n\n" + "\n\n".join(file_data)

    # RAW MODE: go straight through orchestrator (personality hook is handled inside process_user_query)
    if use_raw_gpt:
        logger.info("[Handle Submit] RAW MODE ENABLED – skipping memory and prompt building.")

        response_text, debug_info = await orchestrator.process_user_query(
            user_input=merged_input,
            files=None,
            use_raw_mode=True,
            personality=personality
        )

        # Log the raw mode conversation
        conversation_logger.log_interaction(
            user_input=user_text,  # Log original input without file content for clarity
            assistant_response=response_text,
            metadata={
                'mode': 'raw',
                'files': file_names if file_names else None,
                'personality': personality or getattr(getattr(orchestrator, "personality_manager", None), "current_personality", None),
            }
        )

        # Emit final chunk including debug record for UI tracing
        # Compute token counts for the raw prompt (no system prompt)
        try:
            model_for_tokens = getattr(orchestrator.model_manager, 'get_active_model_name', lambda: None)()
            tm = getattr(orchestrator, 'tokenizer_manager', None)
            prompt_tokens = int(tm.count_tokens(merged_input, model_for_tokens)) if tm else None
        except Exception:
            prompt_tokens = None

        debug_record = {
            'mode': 'raw',
            'query': user_text,
            'prompt': merged_input,
            'system_prompt': None,
            'response': response_text,
            'model': getattr(orchestrator.model_manager, 'get_active_model_name', lambda: None)(),
            'prompt_tokens': prompt_tokens,
            'system_tokens': 0,
            'total_tokens': (prompt_tokens or 0),
        }
        yield {"role": "assistant", "content": response_text, "debug": debug_record}
        return

    # ENHANCED MODE: build prompt first via orchestrator.prepare_prompt (do NOT pass personality here)
    full_prompt, system_prompt = await orchestrator.prepare_prompt(
        user_input=user_text,
        files=files,
        use_raw_mode=False  # enhanced mode
    )

    logger.debug(f"[Handle Submit] Final prompt being passed to model:\n{full_prompt}")

    final_output = ""
    try:
        logger.debug(
            "[🔍 FINAL MESSAGE PAYLOAD TO OPENAI]:\n" +
            json.dumps(
                [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': full_prompt}],
                indent=2
            )
        )

        use_bestof = False
        try:
            qinfo = analyze_query(user_text)
            use_bestof = bool(
                ENABLE_BEST_OF and (
                    (qinfo.is_question and qinfo.token_count >= BEST_OF_MIN_TOKENS)
                    or (not BEST_OF_MIN_QUESTION)
                )
            )
        except Exception:
            use_bestof = bool(ENABLE_BEST_OF)

        model_name = orchestrator.model_manager.get_active_model_name()
        # Token counts for debug trace
        try:
            tm = getattr(orchestrator, 'tokenizer_manager', None)
            prompt_tokens = int(tm.count_tokens(full_prompt, model_name)) if tm else None
            system_tokens = int(tm.count_tokens(system_prompt or '', model_name)) if tm else 0
            total_tokens = (prompt_tokens or 0) + (system_tokens or 0)
        except Exception:
            prompt_tokens = None
            system_tokens = None
            total_tokens = None
        bestof_model = BEST_OF_MODEL or model_name

        if use_bestof:
            # Non-streaming best-of: emit at once
            best = await orchestrator.response_generator.generate_best_of(
                prompt=full_prompt,
                model_name=bestof_model,
                system_prompt=system_prompt,
                question_text=user_text,
                context_hint=full_prompt,
                n=BEST_OF_N,
                temps=tuple(BEST_OF_TEMPS) if isinstance(BEST_OF_TEMPS, (list, tuple)) else (0.2, 0.7),
                max_tokens=BEST_OF_MAX_TOKENS,
            )
            final_output = best
            yield {"role": "assistant", "content": final_output}
        else:
            # Default streaming path
            async for chunk in orchestrator.response_generator.generate_streaming_response(
                prompt=full_prompt,
                model_name=model_name,
                system_prompt=system_prompt
            ):
                final_output = smart_join(final_output, chunk)
                yield {"role": "assistant", "content": final_output}

    except Exception as e:
        logger.error(f"[HANDLE_SUBMIT] Streaming error: {e}")
        error_message = f"⚠️ Streaming error: {str(e)}"

        # Log error conversation
        conversation_logger.log_interaction(
            user_input=user_text,
            assistant_response=error_message,
            metadata={
                'error': str(e),
                'mode': 'enhanced',
                'files': file_names if file_names else None
            }
        )

        yield {"role": "assistant", "content": error_message}

    finally:
        if final_output and len(user_text.strip()) > 0:
            # Log successful conversation
            conversation_logger.log_interaction(
                user_input=user_text,  # Log original input for clarity
                assistant_response=final_output,
                metadata={
                    'mode': 'enhanced',
                    'files': file_names if file_names else None,
                    'personality': personality or getattr(getattr(orchestrator, "personality_manager", None), "current_personality", None),
                    'topic': getattr(orchestrator, 'current_topic', None)
                }
            )

            # Store in memory system (existing code)
            try:
                logger.info("[HANDLE_SUBMIT] Storing interaction in memory...")
                tags = [f"topic:{getattr(orchestrator, 'current_topic', 'general') or 'general'}", "topic:general"]
                # Ensure corpus capacity is generous during testing (override at runtime)
                try:
                    cm = getattr(getattr(orchestrator, "memory_system", None), "corpus_manager", None)
                    if cm and hasattr(cm, "max_entries"):
                        # Default test cap to 5000 if not set via env
                        import os as _os
                        cm.max_entries = int(_os.getenv("CORPUS_MAX_ENTRIES", "5000"))
                except Exception:
                    pass
                await orchestrator.memory_system.store_interaction(
                    query=merged_input,
                    response=final_output,
                    tags=tags
                )
                logger.info("[HANDLE_SUBMIT] Interaction successfully stored.")

                # No mid-session consolidation: summaries are generated at shutdown
            except Exception as e:
                logger.error(f"[HANDLE_SUBMIT] Failed to store interaction: {e}")

            # Emit a final debug record so UI can display Query → Prompt → Response
            try:
                debug_record = {
                    'mode': 'enhanced',
                    'query': user_text,
                    'prompt': full_prompt,
                    'system_prompt': system_prompt,
                    'response': final_output,
                    'model': getattr(orchestrator.model_manager, 'get_active_model_name', lambda: None)(),
                    'prompt_tokens': prompt_tokens,
                    'system_tokens': system_tokens,
                    'total_tokens': total_tokens,
                }
                yield {"role": "assistant", "content": final_output, "debug": debug_record}
            except Exception:
                # If yielding here fails (e.g., consumer closed), just continue silently
                pass
