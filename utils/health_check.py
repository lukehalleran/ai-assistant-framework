"""
# utils/health_check.py

Module Contract
- Purpose: Health check endpoint for Docker/Kubernetes deployments. Provides lightweight service health checks without triggering expensive LLM operations.
- Inputs:
  - get_health_status(orchestrator=None) → Dict[str, Any]: Check application health
  - add_health_endpoint(app, orchestrator=None): Register /health endpoint on FastAPI app
- Outputs:
  - Health status dict with status ("healthy"/"degraded"), timestamp, version, and check results
  - HTTP 200 for healthy, 503 for degraded/unhealthy
- Key checks:
  - Corpus file existence (CORPUS_FILE path)
  - ChromaDB directory existence (CHROMA_PATH)
  - Orchestrator initialization
  - Memory system accessibility
  - API key configuration (checks presence without revealing value)
- Dependencies:
  - config.app_config (CORPUS_FILE, CHROMA_PATH, VERSION)
  - FastAPI (for endpoint registration)
- Side effects:
  - Registers /health endpoint on provided FastAPI app
  - Logs warnings for degraded checks
- Error handling:
  - Gracefully handles missing config values
  - Returns "degraded" status on check failures
  - Falls back to "unknown" version if config unavailable
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


def get_health_status(orchestrator=None) -> Dict[str, Any]:
    """
    Check basic health of the application.

    Returns a dict with health status and basic checks.
    Does NOT make LLM calls to keep health checks lightweight.

    Args:
        orchestrator: Optional orchestrator instance for deeper checks

    Returns:
        Dict with status, timestamp, version, and check results
    """
    checks = {}
    status = "healthy"

    # 1. Check corpus file exists
    try:
        from config.app_config import CORPUS_FILE
        corpus_path = Path(CORPUS_FILE)
        checks["corpus_file"] = {
            "exists": corpus_path.exists(),
            "path": str(corpus_path)
        }
        if not corpus_path.exists():
            logger.warning(f"Health check: corpus file not found at {corpus_path}")
    except Exception as e:
        checks["corpus_file"] = {"error": str(e)}
        status = "degraded"

    # 2. Check ChromaDB directory exists
    try:
        from config.app_config import CHROMA_PATH
        chroma_path = Path(CHROMA_PATH)
        checks["chroma_db"] = {
            "exists": chroma_path.exists(),
            "path": str(chroma_path)
        }
        if not chroma_path.exists():
            logger.warning(f"Health check: ChromaDB not found at {chroma_path}")
    except Exception as e:
        checks["chroma_db"] = {"error": str(e)}
        status = "degraded"

    # 3. Check if orchestrator is available
    if orchestrator:
        checks["orchestrator"] = {"initialized": True}

        # Check if memory system is accessible
        try:
            if hasattr(orchestrator, 'memory_system') and orchestrator.memory_system:
                checks["memory_system"] = {"initialized": True}
            else:
                checks["memory_system"] = {"initialized": False}
                status = "degraded"
        except Exception as e:
            checks["memory_system"] = {"error": str(e)}
            status = "degraded"
    else:
        checks["orchestrator"] = {"initialized": False}

    # 4. Check API key is set (don't reveal the key)
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        checks["api_key"] = {
            "configured": bool(api_key and len(api_key) > 0)
        }
        if not api_key:
            logger.warning("Health check: OPENAI_API_KEY not set")
            status = "degraded"
    except Exception as e:
        checks["api_key"] = {"error": str(e)}
        status = "degraded"

    # Get version
    try:
        from config.app_config import VERSION
        version = VERSION
    except Exception:
        version = "unknown"

    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "version": version,
        "checks": checks
    }


def add_health_endpoint(app, orchestrator=None):
    """
    Add /health endpoint to a FastAPI app (from Gradio).

    Args:
        app: FastAPI app instance (from gradio demo.launch())
        orchestrator: Optional orchestrator for deeper checks

    Raises:
        ImportError: If FastAPI is not available (shouldn't happen with Gradio)
        Exception: If endpoint registration fails
    """
    try:
        # FastAPI is bundled with Gradio, but import here for clarity
        from fastapi import Response
        import json

        @app.get("/health")
        async def health():
            """Health check endpoint for Docker/Kubernetes"""
            health_data = get_health_status(orchestrator)

            # Return 200 for healthy, 503 for degraded/unhealthy
            status_code = 200 if health_data["status"] == "healthy" else 503

            return Response(
                content=json.dumps(health_data, indent=2),
                media_type="application/json",
                status_code=status_code
            )

        logger.info("Health check endpoint registered at /health")

    except ImportError as e:
        logger.error(f"FastAPI not available (required by Gradio): {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to add health endpoint: {e}")
        raise
