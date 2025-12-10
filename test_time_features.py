#!/usr/bin/env python3
"""
Test script to demonstrate the new time tracking features.
Shows how time deltas appear in the [TIME CONTEXT] header.
"""
import sys
import time
sys.path.insert(0, '/home/lukeh/Daemon_RAG_Agent_working')

from utils.time_manager import TimeManager
from core.prompt.formatter import PromptFormatter

def simulate_conversation_session():
    """Simulate a conversation session with time tracking."""

    print("=" * 60)
    print("TIME TRACKING FEATURE DEMONSTRATION")
    print("=" * 60)

    # Create managers
    tm = TimeManager('data/test_time_demo.json')
    formatter = PromptFormatter(token_manager=None, time_manager=tm)

    # === Message 1: First message of the session ===
    print("\n[SCENARIO 1: First message in new session]")
    print("-" * 60)
    print("Expected: 'Time since last message' should show 'N/A (first message in session)'")
    tm.mark_query_time()
    time_ctx = formatter._get_time_context()
    print(time_ctx)
    print()

    # === Message 2: Reply after 3 seconds ===
    time.sleep(3)
    print("\n[SCENARIO 2: User replies 3 seconds later]")
    print("-" * 60)
    print("Expected: 'Time since last message' should show '3 s'")
    tm.mark_query_time()
    time_ctx = formatter._get_time_context()
    print(time_ctx)
    print()

    # === Message 3: Another reply after 5 seconds ===
    time.sleep(5)
    print("\n[SCENARIO 3: User replies 5 seconds later]")
    print("-" * 60)
    print("Expected: 'Time since last message' should show '5 s'")
    tm.mark_query_time()
    time_ctx = formatter._get_time_context()
    print(time_ctx)
    print()

    # === Session ends ===
    print("\n[SESSION ENDS - User closes the application]")
    print("-" * 60)
    tm.mark_session_end()
    print(f"Session end time recorded: {tm.last_session_end_time}")
    print()

    # === Wait and start new session ===
    time.sleep(4)
    print("\n[SCENARIO 4: User starts new session 4 seconds later]")
    print("-" * 60)
    print("Expected: 'Time since last message' shows 'N/A (first message in session)'")
    print("Expected: 'Time since last session' shows '4 s'")
    tm.mark_query_time()
    time_ctx = formatter._get_time_context()
    print(time_ctx)
    print()

    # === Another message in new session ===
    time.sleep(2)
    print("\n[SCENARIO 5: User sends another message 2 seconds later]")
    print("-" * 60)
    print("Expected: 'Time since last message' shows '2 s'")
    print("Expected: 'Time since last session' shows '6 s' (from original session end)")
    tm.mark_query_time()
    time_ctx = formatter._get_time_context()
    print(time_ctx)
    print()

    print("=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("1. Time since last message - tracks gaps between consecutive messages")
    print("2. Time since last session - tracks when user returns after closing app")
    print("3. Both metrics appear in [TIME CONTEXT] prompt header")
    print("4. First message shows 'N/A' for message delta")
    print("5. Session tracking persists across application restarts")
    print()

if __name__ == "__main__":
    simulate_conversation_session()
