#!/usr/bin/env python3
"""
Simple test script for health check endpoint.

This can be run independently to test the health check logic
without starting the full GUI.
"""

import sys
import json
from utils.health_check import get_health_status


def test_health_check():
    """Test health check function"""
    print("Testing health check function...")
    print("=" * 60)

    # Test without orchestrator (basic checks only)
    health_data = get_health_status(orchestrator=None)

    print(json.dumps(health_data, indent=2))
    print("=" * 60)

    if health_data["status"] == "healthy":
        print("✅ Health check passed!")
        return 0
    elif health_data["status"] == "degraded":
        print("⚠️  Health check degraded (some checks failed)")
        return 1
    else:
        print("❌ Health check failed!")
        return 2


if __name__ == "__main__":
    sys.exit(test_health_check())
