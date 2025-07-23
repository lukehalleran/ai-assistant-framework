# daemon_7_11_25_refactor/core/response_generator.py
from utils.logging_utils import get_logger
import time
from typing import AsyncGenerator
from datetime import datetime
from utils.time_manager import TimeManager
logger = get_logger("response_generator")


class ResponseGenerator:
    """Handles response generation and streaming"""

    def __init__(self, model_manager, time_manager: TimeManager = None):
        self.model_manager = model_manager
        self.time_manager = time_manager or TimeManager()
        self.logger = logger

    async def generate_streaming_response(self,
                                        prompt: str,
                                        model_name: str = None) -> AsyncGenerator[str, None]:
        """
        Generate response with streaming support
        """
        self.logger.debug(f"[GENERATE] Starting async generation with model: {model_name}")
        start_time = time.time()
        self.time_manager.mark_query_time()
        self.logger.debug(f"[TIME] Since last query: {self.time_manager.elapsed_since_last()}")
        self.logger.debug(f"[TIME] Previous response time: {self.time_manager.last_response()}")

        first_token_time = None

        try:
            if model_name:
                self.model_manager.switch_model(model_name)
                logger.info(f"[ModelManager] Active model set to: {self.model_manager.get_model()}")

            # Get the async generator
            response_generator = await self.model_manager.generate_async(prompt)

            # Check if it's an async generator
            if hasattr(response_generator, "__aiter__"):
                buffer = ""
                async for chunk in response_generator:
                    try:
                        # Extract content from ChatCompletionChunk
                        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content') and delta.content:
                                delta_content = delta.content
                            else:
                                delta_content = ""
                        else:
                            delta_content = ""

                        if delta_content:
                            now = time.time()
                            if first_token_time is None:
                                first_token_time = now
                                self.logger.debug(f"[STREAMING] First token arrived after {now - start_time:.2f} seconds")

                            buffer += delta_content

                            # For the mock, just yield word by word
                            if " " in buffer:
                                words = buffer.split(" ")
                                for word in words[:-1]:
                                    if word:
                                        yield word
                                buffer = words[-1] if words[-1] else ""

                    except Exception as e:
                        self.logger.error(f"[STREAMING] Error processing chunk: {e}")
                        continue

                # Yield remaining buffer
                if buffer.strip():
                    yield buffer.strip()
                    end_time = time.time()
                    duration = self.time_manager.measure_response(
                        datetime.fromtimestamp(start_time),
                        datetime.fromtimestamp(end_time)
                    )
                    self.logger.info(f"[TIMING] Full response duration: {duration}")

            else:
                # Handle non-streaming response (synchronous fallback)
                self.logger.debug("[GENERATE] Non-streaming response, handling in fallback mode.")

                # If generate_async returned a regular response object
                if hasattr(response_generator, "choices") and len(response_generator.choices) > 0:
                    if hasattr(response_generator.choices[0], "message"):
                        content = response_generator.choices[0].message.content
                    else:
                        content = str(response_generator)
                else:
                    content = str(response_generator)

                # Simulate streaming by yielding words
                words = content.split()
                for word in words:
                    yield word

        except Exception as e:
            self.logger.error(f"[GENERATE] Error: {type(e).__name__}: {str(e)}")
            yield f"[Streaming Error] {e}"
