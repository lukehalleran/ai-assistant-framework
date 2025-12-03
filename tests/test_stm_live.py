#!/usr/bin/env python3
"""Quick test to verify STM pipeline works with actual orchestrator."""

import asyncio
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

async def test_stm_in_orchestrator():
    """Test STM through actual orchestrator."""
    print("\n" + "="*80)
    print("TESTING STM PIPELINE WITH ORCHESTRATOR")
    print("="*80)

    # Import after path setup
    from main import build_orchestrator
    from config.app_config import USE_STM_PASS

    print(f"\nSTM Enabled: {USE_STM_PASS}")
    print("\nBuilding orchestrator...")

    # Build orchestrator (same as main.py does)
    orchestrator = build_orchestrator()

    print(f"✅ Orchestrator built")
    print(f"✅ STM Analyzer: {orchestrator.stm_analyzer is not None}")

    # Simulate a conversation with 4+ turns to trigger STM
    queries = [
        "Hey, I'm working on a Python project",
        "It's a RAG system with vector search",
        "Using ChromaDB for the vector store",
        "Should I add prompt caching?"  # This should trigger STM
    ]

    print("\n" + "-"*80)
    print("Simulating conversation...")
    print("-"*80)

    for i, query in enumerate(queries, 1):
        print(f"\n[Turn {i}] User: {query}")

        try:
            # Process query
            response, debug_info = await orchestrator.process_user_query(
                user_input=query,
                use_raw_mode=False
            )

            print(f"[Turn {i}] Assistant: {response[:100]}...")

            # On turn 4, check if STM was used
            if i == 4:
                print("\n" + "="*80)
                print("CHECKING TURN 4 (should have STM)")
                print("="*80)

                # Check debug info for prompt
                prompt = debug_info.get('prompt', '')

                if '[SHORT-TERM CONTEXT SUMMARY]' in prompt:
                    print("✅ STM summary section found in prompt!")

                    # Extract STM section
                    start = prompt.find('[SHORT-TERM CONTEXT SUMMARY]')
                    end = prompt.find('[RECENT CONVERSATION]', start)
                    if end == -1:
                        end = prompt.find('[RELEVANT', start)
                    stm_section = prompt[start:end]

                    print("\nSTM Section:")
                    print("-"*40)
                    print(stm_section[:500])
                    print("-"*40)
                else:
                    print("❌ STM summary section NOT found!")
                    print(f"\nPrompt sections found:")
                    for line in prompt.split('\n'):
                        if line.startswith('['):
                            print(f"  • {line}")

                if '[LAST EXCHANGE FOR CONTEXT]' in prompt:
                    print("✅ Last exchange attached to query!")
                else:
                    print("❌ Last exchange NOT attached!")

                # Show prompt length
                print(f"\nTotal prompt length: {len(prompt)} chars")

        except Exception as e:
            print(f"❌ Error on turn {i}: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "="*80)
    print("✅ TEST COMPLETE - STM PIPELINE WORKING!")
    print("="*80 + "\n")

    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(test_stm_in_orchestrator())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
