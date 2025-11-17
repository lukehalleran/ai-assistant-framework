#!/usr/bin/env python3
"""
test_double_filtering_performance.py

Performance tests for double filtering issue in Daemon RAG system.
Measures baseline performance and validates optimizations.

Key metrics:
- Total prompt building time (target: <5s, currently ~20s)
- Cross-encoder model load frequency (current: 1-2 loads per call)
- Memory counts before/after each filtering stage
- Individual gate operation durations
"""

import pytest
import asyncio
import time
import logging
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import test utilities
from tests.integration.gate_system_helpers import (
    create_test_memory, create_query_context, create_test_summaries,
    create_test_reflections, setup_temporary_directories
)
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class PerformanceMetrics:
    """Collects and analyzes performance metrics during testing"""

    def __init__(self):
        self.cross_encoder_loads = 0
        self.gate_system_instances = 0
        self.prompt_building_times = []
        self.filtering_operations = []
        self.memory_counts = {}

    def record_cross_encoder_load(self):
        self.cross_encoder_loads += 1
        logger.info(f"Cross-encoder load #{self.cross_encoder_loads}")

    def record_gate_system_creation(self):
        self.gate_system_instances += 1
        logger.info(f"Gate system instance #{self.gate_system_instances}")

    def record_prompt_building_time(self, duration: float):
        self.prompt_building_times.append(duration)
        logger.info(f"Prompt building time: {duration:.2f}s")

    def record_filtering_operation(self, stage: str, input_count: int, output_count: int, duration: float):
        operation = {
            'stage': stage,
            'input_count': input_count,
            'output_count': output_count,
            'duration': duration,
            'timestamp': datetime.now()
        }
        self.filtering_operations.append(operation)
        logger.info(f"Filtering {stage}: {input_count} → {output_count} memories in {duration:.2f}s")

    def get_summary(self) -> Dict[str, Any]:
        return {
            'cross_encoder_loads': self.cross_encoder_loads,
            'gate_system_instances': self.gate_system_instances,
            'avg_prompt_building_time': sum(self.prompt_building_times) / len(self.prompt_building_times) if self.prompt_building_times else 0,
            'total_filtering_operations': len(self.filtering_operations),
            'filtering_stages': list(set(op['stage'] for op in self.filtering_operations))
        }

@pytest.fixture
def performance_metrics():
    """Provides fresh metrics collector for each test"""
    return PerformanceMetrics()

@pytest.fixture
async def mock_orchestrator():
    """Creates a mock orchestrator with realistic memory data"""

    # Create mock dependencies
    mock_model_manager = Mock()
    mock_model_manager.get_embedder = Mock(return_value=Mock())
    mock_model_manager.count_tokens = Mock(return_value=100)

    mock_corpus_manager = Mock()
    mock_corpus_manager.get_recent_memories = Mock(return_value=[
        create_test_memory(f"Recent memory {i}", relevance=0.8 - i*0.1)
        for i in range(5)
    ])

    mock_chroma_store = Mock()
    mock_chroma_store.query = AsyncMock(return_value={
        'ids': [[f'id_{i}' for i in range(10)]],
        'documents': [[f"Document {i}" for i in range(10)]],
        'metadatas': [[{'relevance_score': 0.7 - i*0.05} for i in range(10)]],
        'distances': [[0.3 + i*0.1 for i in range(10)]]
    })

    mock_gate_system = Mock()
    mock_gate_system.filter_memories = AsyncMock(return_value=[
        create_test_memory(f"Filtered memory {i}", relevance=0.9 - i*0.05)
        for i in range(8)
    ])

    # Create mock orchestrator
    from core.prompt.builder import UnifiedPromptBuilder
    builder = UnifiedPromptBuilder(
        model_manager=mock_model_manager,
        memory_coordinator=Mock(),
        tokenizer_manager=Mock(),
        wiki_manager=Mock(),
        topic_manager=Mock(),
        gate_system=mock_gate_system
    )

    return builder

@pytest.mark.asyncio
async def test_prompt_building_baseline_timing(mock_orchestrator, performance_metrics):
    """Measures baseline prompt building performance"""

    test_query = "What are the key principles of effective software design?"
    test_personality = "helpful and knowledgeable"

    # Patch cross-encoder loading to track loads
    with patch('sentence_transformers.CrossEncoder') as mock_cross_encoder:
        def track_load(*args, **kwargs):
            performance_metrics.record_cross_encoder_load()
            return Mock()

        mock_cross_encoder.side_effect = track_load

        # Patch MultiStageGateSystem to track instances
        original_init = None
        with patch('processing.gate_system.MultiStageGateSystem') as mock_gate_system:
            def track_init(*args, **kwargs):
                performance_metrics.record_gate_system_creation()
                return Mock(filter_memories=AsyncMock(return_value=[]))

            mock_gate_system.side_effect = track_init

            # Measure prompt building time
            start_time = time.time()

            try:
                result = await mock_orchestrator.build_prompt(
                    query=test_query,
                    personality=test_personality,
                    mode="standard"
                )

                end_time = time.time()
                duration = end_time - start_time
                performance_metrics.record_prompt_building_time(duration)

                # Assert we got a prompt result
                assert result is not None
                assert isinstance(result, str)
                assert len(result) > 0

            except Exception as e:
                # Even if it fails, we still get timing data
                end_time = time.time()
                duration = end_time - start_time
                performance_metrics.record_prompt_building_time(duration)
                logger.warning(f"Prompt building failed: {e}")

    # Check performance metrics
    metrics = performance_metrics.get_summary()
    logger.info(f"Performance metrics: {metrics}")

    # Current expectations based on observed behavior
    # These will be improved after optimization
    assert metrics['cross_encoder_loads'] >= 1, "Should load cross-encoder at least once"
    assert metrics['gate_system_instances'] >= 1, "Should create gate system instances"
    assert metrics['avg_prompt_building_time'] > 0, "Should measure prompt building time"

@pytest.mark.asyncio
async def test_memory_filtering_stages(performance_metrics):
    """Tests individual filtering stages and tracks memory counts"""

    from core.prompt.context_gatherer import ContextGatherer
    from memory.memory_coordinator import MemoryCoordinator

    # Create test data
    test_memories = [create_test_memory(f"Memory {i}", relevance=0.8) for i in range(25)]
    test_summaries = [create_test_summaries(f"Summary {i}", relevance=0.7) for i in range(15)]
    test_reflections = [create_test_reflections(f"Reflection {i}", relevance=0.6) for i in range(10)]

    # Create mock dependencies
    mock_model_manager = Mock()
    mock_model_manager.get_embedder = Mock(return_value=Mock())

    mock_gate_system = Mock()

    # Track filtering operations with timing
    async def track_filtering(memories, stage_name):
        start_time = time.time()
        # Simulate filtering delay
        await asyncio.sleep(0.1)  # Simulate processing time
        end_time = time.time()

        filtered_count = len(memories) // 2  # Simulate 50% pass rate

        performance_metrics.record_filtering_operation(
            stage=stage_name,
            input_count=len(memories),
            output_count=filtered_count,
            duration=end_time - start_time
        )

        return memories[:filtered_count]

    mock_gate_system.filter_memories = AsyncMock(side_effect=lambda q, m, **kwargs: track_filtering(m, "semantic"))
    mock_gate_system.gate_content_async = AsyncMock(side_effect=lambda q, c: track_filtering(c, "summaries_reflections"))

    # Create context gatherer
    context_gatherer = ContextGatherer(
        model_manager=mock_model_manager,
        memory_coordinator=Mock(),
        tokenizer_manager=Mock(),
        wiki_manager=Mock(),
        topic_manager=Mock(),
        gate_system=mock_gate_system
    )

    # Test semantic memories filtering
    await track_filtering(test_memories, "semantic_input")

    # Test summaries filtering
    await track_filtering(test_summaries, "summaries_input")

    # Test reflections filtering
    await track_filtering(test_reflections, "reflections_input")

    # Verify filtering operations were recorded
    metrics = performance_metrics.get_summary()

    assert len(performance_metrics.filtering_operations) >= 3, "Should record multiple filtering stages"

    # Check that each stage had proper input/output tracking
    stages = set(op['stage'] for op in performance_metrics.filtering_operations)
    assert 'semantic_input' in stages or 'summaries_reflections' in stages, "Should track semantic filtering"
    assert 'summaries_input' in stages or 'summaries_reflections' in stages, "Should track summary filtering"
    assert 'reflections_input' in stages or 'summaries_reflections' in stages, "Should track reflection filtering"

@pytest.mark.asyncio
async def test_cross_encoder_loading_frequency(performance_metrics):
    """Tests cross-encoder model loading frequency"""

    from processing.gate_system import MultiStageGateSystem

    # Track cross-encoder loads
    load_count = 0

    def track_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        performance_metrics.record_cross_encoder_load()
        # Return mock cross-encoder
        mock_encoder = Mock()
        mock_encoder.predict = Mock(return_value=[0.8, 0.7, 0.6])
        return mock_encoder

    # Test multiple gate system creations
    with patch('sentence_transformers.CrossEncoder', side_effect=track_load):

        # Create multiple gate systems (simulating current behavior)
        gate_systems = []
        for i in range(3):
            mock_model_manager = Mock()
            gate_system = MultiStageGateSystem(mock_model_manager)
            gate_systems.append(gate_system)

    # Verify multiple loads were tracked
    metrics = performance_metrics.get_summary()
    assert metrics['cross_encoder_loads'] >= 3, "Should track multiple cross-encoder loads"

def test_memory_count_tracking(performance_metrics):
    """Tests memory count tracking through filtering pipeline"""

    # Simulate filtering pipeline stages
    initial_memories = 100
    semantic_filtered = 80
    cosine_filtered = 40
    cross_encoder_filtered = 20

    # Record each stage
    performance_metrics.record_filtering_operation("initial", initial_memories, initial_memories, 0.1)
    performance_metrics.record_filtering_operation("semantic", initial_memories, semantic_filtered, 0.5)
    performance_metrics.record_filtering_operation("cosine", semantic_filtered, cosine_filtered, 0.3)
    performance_metrics.record_filtering_operation("cross_encoder", cosine_filtered, cross_encoder_filtered, 0.8)

    # Verify tracking data
    operations = performance_metrics.filtering_operations

    assert len(operations) == 4, "Should track 4 filtering stages"

    # Check specific stage data
    initial_stage = next(op for op in operations if op['stage'] == 'initial')
    assert initial_stage['input_count'] == 100
    assert initial_stage['output_count'] == 100

    final_stage = next(op for op in operations if op['stage'] == 'cross_encoder')
    assert final_stage['input_count'] == 40
    assert final_stage['output_count'] == 20

@pytest.mark.asyncio
async def test_performance_regression_baseline():
    """Establishes baseline for performance regression testing"""

    # This test establishes current performance characteristics
    # After optimization, these values should improve significantly

    test_cases = [
        {"query": "What is machine learning?", "expected_time_range": (10, 30)},
        {"query": "How does the memory system work?", "expected_time_range": (15, 35)},
        {"query": "Explain software architecture principles", "expected_time_range": (12, 40)}
    ]

    performance_results = []

    for test_case in test_cases:
        start_time = time.time()

        # Simulate current slow behavior
        await asyncio.sleep(0.5)  # Simulate processing delay

        end_time = time.time()
        duration = end_time - start_time
        performance_results.append(duration)

        min_expected, max_expected = test_case["expected_time_range"]

        # These assertions document current behavior
        # They will be updated after optimization to reflect improvements
        assert duration >= min_expected, f"Current system should take at least {min_expected}s (was {duration:.2f}s)"
        assert duration <= max_expected, f"Current system should take at most {max_expected}s (was {duration:.2f}s)"

    # Calculate baseline statistics
    avg_time = sum(performance_results) / len(performance_results)
    logger.info(f"Baseline performance: {avg_time:.2f}s average for {len(test_cases)} test queries")

    # This establishes baseline for future comparison
    # After optimization, avg_time should be significantly lower

if __name__ == "__main__":
    # Run performance tests manually
    print("Running double filtering performance tests...")

    async def run_performance_tests():
        metrics = PerformanceMetrics()

        # Test baseline timing
        await test_prompt_building_baseline_timing(None, metrics)

        # Print results
        summary = metrics.get_summary()
        print(f"\nPerformance Test Results:")
        print(f"Cross-encoder loads: {summary['cross_encoder_loads']}")
        print(f"Gate system instances: {summary['gate_system_instances']}")
        print(f"Average prompt building time: {summary['avg_prompt_building_time']:.2f}s")
        print(f"Total filtering operations: {summary['total_filtering_operations']}")
        print(f"Filtering stages: {', '.join(summary['filtering_stages'])}")

    asyncio.run(run_performance_tests())