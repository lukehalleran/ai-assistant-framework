#tests/test_summaries.py
import asyncio
from memory.corpus_manager import CorpusManager
from utils.logging_utils import get_logger

logger = get_logger("test_summaries")

async def test_summary_integration():
    corpus_manager = CorpusManager("test_corpus.json")

    logger.info("Adding 25 test entries...")
    for i in range(25):
        corpus_manager.add_entry(
            query=f"What is {i}?",
            response=f"It is {i * 2}",
            tags=[f"test"]
        )

    summaries = corpus_manager.get_summaries()
    logger.info(f"Auto-created summaries: {len(summaries)}")

    logger.info("Manually creating 10-entry summary...")
    summary = corpus_manager.create_summary_now(10)
    if summary:
        logger.info(f"Manual summary created:\n{summary[:200]}...")

    logger.info("Inspecting all summaries:")
    for i, s in enumerate(corpus_manager.get_summaries(count=5)):
        logger.info(f"[{i}] Tags: {s.get('tags')}, Preview: {s.get('response', '')[:100]}...")

if __name__ == "__main__":
    asyncio.run(test_summary_integration())
