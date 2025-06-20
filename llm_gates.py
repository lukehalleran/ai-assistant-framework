import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import asyncio
import ModelManager

@dataclass
class GateResult:
    relevant: bool
    confidence: float
    reasoning: str
    filtered_content: str = None

class LLMGateSystem:
    def __init__(self, model_manager, fast_model="gpt-3.5-turbo"):
        self.model_manager = model_manager
        self.fast_model = fast_model
        self.cache = {}

    def is_meta_query(self, query: str) -> bool:
        meta_keywords = ["memory", "memories", "what do you know", "how much do you remember", "what have you stored", "what do you recall"]
        return any(kw in query.lower() for kw in meta_keywords)
        def get_top_discussion_topics(self, top_n: int = 5) -> List[Tuple[str, int]]:
            from collections import Counter
            all_content = [m.content.lower() for m in self.memories.values()]
            tokenized = [word for text in all_content for word in text.split()]
            common = Counter(tokenized).most_common()
            return [(word, count) for word, count in common if word.isalpha()][:top_n]

    def create_relevance_prompt(self, query: str, content: str, content_type: str) -> str:
        meta_hint = "This is a meta query about the assistant's memory." if self.is_meta_query(query) else ""
        return f"""Determine if this {content_type} is relevant to the user's query.

User Query: "{query}"

{content_type.capitalize()} Content:
{content[:500]}...

Respond in JSON format:
{{
    "relevant": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "key_points": ["point1", "point2"]
}}

Guidelines:
- If the user's query is about the assistant itself (e.g., memory count, past discussions, what you know about them), treat content related to memory, interaction history, or system behavior as potentially relevant.
- Prioritize conceptual and structural connections, not just exact topic matches.
- Focus on semantic relevance, not just keyword overlap.
{meta_hint}
"""

    def _clean_json_response(self, text: str) -> str:
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    async def gate_content_async(self, query: str, content: str, content_type: str) -> GateResult:
        cache_key = f"{query[:50]}:{content[:100]}:{content_type}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = self.create_relevance_prompt(query, content, content_type)
        try:
            response = await asyncio.to_thread(
                self.model_manager.generate,
                prompt,
                model_name=self.fast_model,
                max_tokens=150,
                temperature=0.1
            )

            # Handle empty or None responses
            if not response or not response.strip():
                print(f"[Gate Warning] Empty response for {content_type}")
                return GateResult(relevant=True, confidence=0.5, reasoning="Empty response, including by default")

            try:
                cleaned_response = self._clean_json_response(response)
                print(f"[Gate DEBUG] Cleaned model response:\n{cleaned_response}")

                # Additional validation
                if not cleaned_response or cleaned_response == "None":
                    return GateResult(relevant=True, confidence=0.5, reasoning="Invalid response format")

                result_data = json.loads(cleaned_response)

            except json.JSONDecodeError as json_err:
                print(f"[Gate Warning] Invalid JSON response: {response[:100]}... Error: {json_err}")
                response_lower = response.lower()
                if "not relevant" in response_lower or "irrelevant" in response_lower:
                    return GateResult(relevant=False, confidence=0.3, reasoning="Inferred from response")
                else:
                    return GateResult(relevant=True, confidence=0.5, reasoning="Could not parse response")

            result = GateResult(
                relevant=result_data.get("relevant", False),
                confidence=result_data.get("confidence", 0.0),
                reasoning=result_data.get("reasoning", ""),
                filtered_content=self._extract_key_points(content, result_data.get("key_points", []))
            )
            self.cache[cache_key] = result
            return result

        except Exception as e:
            print(f"[Gate Error] {type(e).__name__}: {e}")
            return GateResult(relevant=True, confidence=0.5, reasoning="Gate failed, including by default")

    def _extract_key_points(self, content: str, key_points: List[str]) -> str:
        if not key_points:
            return content[:300]
        sentences = content.split('. ')
        relevant_sentences = [
            sentence for sentence in sentences
            if any(point.lower() in sentence.lower() for point in key_points)
        ]
        return '. '.join(relevant_sentences[:5]) or content[:300]


class MultiStageGateSystem:
    def __init__(self, model_manager):
        self.gate_system = LLMGateSystem(model_manager)
    def _clean_json_response(self, response: str) -> str:
        response = response.strip()
        if response.startswith("```json"):
            response = response[len("```json"):].strip()
        if response.endswith("```"):
            response = response[:-3].strip()
        return response


    async def filter_memories(self, query: str, memories: List[Dict]) -> List[Dict]:
        if not memories:
            return []

        tasks = []
        for memory in memories:
            content = f"User: {memory.get('query', '')}\nAssistant: {memory.get('response', '')}"
            task = self.gate_system.gate_content_async(query, content, "memory")
            tasks.append((memory, task))

        filtered_memories = []
        for memory, task in tasks:
            try:
                result = await task
                if result.relevant and result.confidence > 0.6:
                    memory['relevance_score'] = result.confidence
                    memory['filtered_content'] = result.filtered_content
                    filtered_memories.append(memory)
            except Exception as e:
                print(f"[Memory Filter Error] {e}")
                # Include with default score on error
                memory['relevance_score'] = 0.5
                filtered_memories.append(memory)

        return sorted(filtered_memories, key=lambda x: x['relevance_score'], reverse=True)

    async def filter_wiki_content(self, query: str, wiki_content: str) -> Tuple[bool, str]:
        if not wiki_content:
            return False, ""

        try:
            result = await self.gate_system.gate_content_async(query, wiki_content, "wikipedia")
            if result.relevant and result.confidence > 0.7:
                return True, result.filtered_content or wiki_content[:500]
            return False, ""
        except Exception as e:
            print(f"[Wiki Filter Error] {e}")
            return False, ""

    async def filter_semantic_chunks(self, query: str, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return []

        title_tasks = []
        for chunk in chunks:
            prompt = f"Is '{chunk['title']}' relevant to: '{query}'? Answer yes/no."
            task = asyncio.create_task(
                asyncio.to_thread(
                    self.gate_system.model_manager.generate,
                    prompt,
                    model_name=self.gate_system.fast_model,
                    max_tokens=10,
                    temperature=0
                )
            )
            title_tasks.append((chunk, task))

        title_relevant_chunks = []
        for chunk, task in title_tasks:
            try:
                response = await task
                if response and "yes" in response.lower():
                    title_relevant_chunks.append(chunk)
            except Exception as e:
                print(f"[Title Filter Error] {e}")
                # Include chunk on error
                title_relevant_chunks.append(chunk)

        if title_relevant_chunks:
            content_tasks = []
            for chunk in title_relevant_chunks[:10]:
                task = self.gate_system.gate_content_async(query, chunk['text'], "article_chunk")
                content_tasks.append((chunk, task))

            final_chunks = []
            for chunk, task in content_tasks:
                try:
                    result = await task
                    if result.relevant:
                        chunk['relevance_score'] = result.confidence
                        chunk['key_content'] = result.filtered_content
                        final_chunks.append(chunk)
                except Exception as e:
                    print(f"[Content Filter Error] {e}")
                    # Include with default score
                    chunk['relevance_score'] = 0.5
                    final_chunks.append(chunk)

            return sorted(final_chunks, key=lambda x: x['relevance_score'], reverse=True)[:5]

        return []

class GatedPromptBuilder:
    def __init__(self, prompt_builder, model_manager):
        self.prompt_builder = prompt_builder
        self.gate_system = MultiStageGateSystem(model_manager)

    async def build_gated_prompt(self, user_input: str, context_sources: Dict) -> str:
        filtered_context = {}

        try:
            filtered_context["memories"] = await self.gate_system.filter_memories(
                user_input, context_sources.get("memories", [])
            )
        except Exception as e:
            print(f"[Gated Prompt - Memory Error] {e}")
            filtered_context["memories"] = context_sources.get("memories", [])[:5]  # Fallback to first 5

        try:
            include_wiki, filtered_wiki = await self.gate_system.filter_wiki_content(
                user_input, context_sources.get("wiki_snippet", "")
            )
            if include_wiki:
                filtered_context["wiki_snippet"] = filtered_wiki
        except Exception as e:
            print(f"[Gated Prompt - Wiki Error] {e}")
            filtered_context["wiki_snippet"] = ""

        try:
            filtered_context["semantic_chunks"] = await self.gate_system.filter_semantic_chunks(
                user_input, context_sources.get("semantic_chunks", [])
            )
        except Exception as e:
            print(f"[Gated Prompt - Semantic Error] {e}")
            filtered_context["semantic_chunks"] = []

        return self.prompt_builder.build_prompt(
            model_name=self.prompt_builder.tokenizer_manager.active_model_name,
            user_input=user_input,
            memories=filtered_context.get("memories", []),
            summaries=context_sources.get("summaries", []),
            dreams=[],
            wiki_snippet=filtered_context.get("wiki_snippet", ""),
            semantic_snippet=filtered_context.get("semantic_chunks", []),
            time_context=context_sources.get("time_context"),
            is_api=True
        )
