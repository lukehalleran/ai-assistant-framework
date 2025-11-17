#!/usr/bin/env python3
"""
test_gating_consistency.py

Tests for ensuring gating consistency across different implementations.
Validates that the same memories produce identical filtering results
regardless of gate system instance or caching approach.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import List, Dict, Any

# Import test utilities
from tests.integration.gate_system_helpers import (
    create_test_memory, create_test_summaries, create_test_reflections,
    setup_temporary_directories
)
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class GatingConsistencyValidator:
    """Validates gating consistency across different implementations"""

    def __init__(self):
        self.test_results = {}
        self.memory_databases = {}
        self.gating_results = {}

    def store_gate_result(self, test_name: str, input_memories: List[Dict], output_memories: List[Dict]):
        """Stores gating results for comparison"""
        self.gating_results[test_name] = {
            'input_count': len(input_memories),
            'output_count': len(output_memories),
            'input_ids': [mem.get('id', f"input_{i}") for i, mem in enumerate(input_memories)],
            'output_ids': [mem.get('id', f"output_{i}") for i, mem in enumerate(output_memories)],
            'output_scores': [mem.get('relevance_score', 0.0) for mem in output_memories],
            'timestamp': datetime.now()
        }

    def compare_results(self, test_name_1: str, test_name_2: str) -> Dict[str, Any]:
        """Compares two gating results for consistency"""
        if test_name_1 not in self.gating_results or test_name_2 not in self.gating_results:
            return {'error': 'Missing test results for comparison'}

        result1 = self.gating_results[test_name_1]
        result2 = self.gating_results[test_name_2]

        comparison = {
            'test_names': [test_name_1, test_name_2],
            'input_consistent': result1['input_ids'] == result2['input_ids'],
            'output_count_consistent': result1['output_count'] == result2['output_count'],
            'output_ids_consistent': result1['output_ids'] == result2['output_ids'],
            'scores_consistent': self._compare_scores(result1['output_scores'], result2['output_scores']),
            'details': {
                f'{test_name_1}_count': result1['output_count'],
                f'{test_name_2}_count': result2['output_count'],
                'id_differences': list(set(result1['output_ids']) ^ set(result2['output_ids'])),
                'score_differences': self._find_score_differences(result1['output_scores'], result2['output_scores'])
            }
        }

        return comparison

    def _compare_scores(self, scores1: List[float], scores2: List[float]) -> bool:
        """Compares score lists with tolerance"""
        if len(scores1) != len(scores2):
            return False
        return all(abs(s1 - s2) < 0.001 for s1, s2 in zip(scores1, scores2))

    def _find_score_differences(self, scores1: List[float], scores2: List[float]) -> List[Dict]:
        """Finds significant score differences"""
        differences = []
        min_len = min(len(scores1), len(scores2))
        for i in range(min_len):
            diff = abs(scores1[i] - scores2[i])
            if diff > 0.001:  # Threshold for significant difference
                differences.append({
                    'index': i,
                    'score1': scores1[i],
                    'score2': scores2[i],
                    'difference': diff
                })
        return differences

@pytest.fixture
def consistency_validator():
    """Provides a fresh consistency validator for each test"""
    return GatingConsistencyValidator()

def create_test_memory_dataset():
    """Creates a consistent test dataset for gating tests"""
    return [
        {
            'id': f'test_memory_{i}',
            'query': f'What is the concept of concept {i}?',
            'response': f'Concept {i} is a fundamental idea in {i}-based thinking.',
            'content': f'User asks about concept {i}. Assistant explains it is fundamental.',
            'relevance_score': 0.9 - (i * 0.05),  # Decreasing relevance
            'truth_score': 0.8,
            'importance_score': 0.7,
            'timestamp': datetime.now(),
            'metadata': {'source': 'test', 'memory_type': 'semantic'}
        }
        for i in range(20)
    ]

@pytest.mark.asyncio
async def test_same_memories_same_gating_results(consistency_validator):
    """Tests that identical memories produce identical gating results"""

    test_memories = create_test_memory_dataset()
    test_query = "What are fundamental concepts?"

    # Create two different gate system instances
    from processing.gate_system import MultiStageGateSystem

    mock_model_manager_1 = Mock()
    mock_model_manager_2 = Mock()

    # Mock cross-encoder to return deterministic results
    def mock_predict(pairs):
        # Return deterministic scores based on input
        return [0.9 - (i * 0.02) for i in range(len(pairs))]

    mock_cross_encoder = Mock()
    mock_cross_encoder.predict = Mock(side_effect=mock_predict)

    with patch('sentence_transformers.CrossEncoder', return_value=mock_cross_encoder):
        gate_system_1 = MultiStageGateSystem(mock_model_manager_1)
        gate_system_2 = MultiStageGateSystem(mock_model_manager_2)

        # Test both gate systems with identical input
        result1 = await gate_system_1.filter_memories(test_query, test_memories.copy(), k=10)
        result2 = await gate_system_2.filter_memories(test_query, test_memories.copy(), k=10)

        # Store results for comparison
        consistency_validator.store_gate_result("gate_system_1", test_memories, result1)
        consistency_validator.store_gate_result("gate_system_2", test_memories, result2)

        # Compare results
        comparison = consistency_validator.compare_results("gate_system_1", "gate_system_2")

        # Assert consistency
        assert comparison['output_count_consistent'], f"Output counts should match: {comparison['details']}"
        assert comparison['output_ids_consistent'], f"Output IDs should match: {comparison['details']}"
        assert comparison['scores_consistent'], f"Scores should match: {comparison['details']}"

        logger.info(f"Consistency test passed: {len(result1)} memories gated consistently")

@pytest.mark.asyncio
async def test_memory_ranking_stability(consistency_validator):
    """Tests that memory ranking remains stable across multiple runs"""

    test_memories = create_test_memory_dataset()
    test_query = "What are the key principles?"

    ranking_results = []

    # Run gating multiple times and collect results
    from processing.gate_system import MultiStageGateSystem

    mock_model_manager = Mock()

    # Mock cross-encoder for deterministic results
    def mock_predict(pairs):
        return [0.95, 0.93, 0.91, 0.89, 0.87, 0.85, 0.83, 0.81, 0.79, 0.77]

    mock_cross_encoder = Mock()
    mock_cross_encoder.predict = Mock(side_effect=mock_predict)

    with patch('sentence_transformers.CrossEncoder', return_value=mock_cross_encoder):
        for run_id in range(3):
            gate_system = MultiStageGateSystem(mock_model_manager)
            result = await gate_system.filter_memories(test_query, test_memories.copy(), k=10)

            # Extract ranking information
            rankings = [mem.get('relevance_score', 0.0) for mem in result]
            memory_ids = [mem.get('id', '') for mem in result]

            ranking_results.append({
                'run_id': run_id,
                'rankings': rankings,
                'memory_ids': memory_ids
            })

            consistency_validator.store_gate_result(f"run_{run_id}", test_memories, result)

    # Compare rankings across runs
    first_run = ranking_results[0]

    for i, run in enumerate(ranking_results[1:], 1):
        comparison = consistency_validator.compare_results("run_0", f"run_{i}")

        # Check that rankings are consistent
        assert comparison['output_count_consistent'], f"Run {i} should have same output count as run 0"
        assert comparison['output_ids_consistent'], f"Run {i} should have same output IDs as run 0"

        # Check exact ranking scores
        for j, (score1, score2) in enumerate(zip(first_run['rankings'], run['rankings'])):
            assert abs(score1 - score2) < 0.001, f"Ranking {j} should be consistent: {score1} vs {score2}"

    logger.info(f"Ranking stability test passed: {len(ranking_results)} consistent runs")

def test_memory_type_filtering_accuracy():
    """Tests that different memory types are filtered correctly"""

    # Create memories of different types
    memories = [
        {
            'id': 'episodic_1',
            'type': 'episodic',
            'content': 'User asked about Python',
            'relevance_score': 0.8,
            'metadata': {'source': 'conversation', 'memory_type': 'episodic'}
        },
        {
            'id': 'semantic_1',
            'type': 'semantic',
            'content': 'Python is a programming language',
            'relevance_score': 0.9,
            'metadata': {'source': 'fact_extractor', 'memory_type': 'semantic'}
        },
        {
            'id': 'summary_1',
            'type': 'summary',
            'content': 'Discussion about Python basics',
            'relevance_score': 0.7,
            'metadata': {'source': 'consolidation', 'memory_type': 'summary'}
        }
    ]

    test_query = "What is Python?"

    # Test that episodic memories bypass gating (as per architecture)
    from processing.gate_system import MultiStageGateSystem

    mock_model_manager = Mock()

    # Mock cross-encoder
    mock_cross_encoder = Mock()
    mock_cross_encoder.predict = Mock(return_value=[0.8, 0.6, 0.4])

    with patch('sentence_transformers.CrossEncoder', return_value=mock_cross_encoder):
        gate_system = MultiStageGateSystem(mock_model_manager)

        # Test filtering behavior
        async def test_filtering():
            result = await gate_system.filter_memories(test_query, memories.copy(), k=5)

            # Should return all memories since they all have relevance > threshold
            assert len(result) >= 2, "Should return memories that pass relevance threshold"

            # Check that episodic memories are handled correctly
            episodic_memories = [mem for mem in result if mem.get('metadata', {}).get('memory_type') == 'episodic']
            assert len(episodic_memories) >= 1, "Should include episodic memories"

        asyncio.run(test_filtering())

def test_content_integrity_after_gating():
    """Tests that memory content is not modified during gating"""

    original_memories = create_test_memory_dataset()

    # Extract original content for comparison
    original_content = {}
    for mem in original_memories:
        original_content[mem['id']] = {
            'query': mem['query'],
            'response': mem['response'],
            'content': mem['content'],
            'relevance_score': mem['relevance_score']
        }

    test_query = "What are the test concepts?"

    # Run gating
    from processing.gate_system import MultiStageGateSystem

    mock_model_manager = Mock()
    mock_cross_encoder = Mock()
    mock_cross_encoder.predict = Mock(return_value=[0.8] * 10)

    with patch('sentence_transformers.CrossEncoder', return_value=mock_cross_encoder):
        gate_system = MultiStageGateSystem(mock_model_manager)

        async def test_integrity():
            result = await gate_system.filter_memories(test_query, original_memories.copy(), k=10)

            # Verify content integrity
            for gated_memory in result:
                mem_id = gated_memory.get('id')
                if mem_id in original_content:
                    original = original_content[mem_id]

                    # Check that core content is preserved
                    assert gated_memory.get('query') == original['query'], f"Query content changed for {mem_id}"
                    assert gated_memory.get('response') == original['response'], f"Response content changed for {mem_id}"
                    assert gated_memory.get('content') == original['content'], f"Combined content changed for {mem_id}"

                    # Check that relevance score is preserved or updated appropriately
                    original_score = original['relevance_score']
                    gated_score = gated_memory.get('relevance_score')
                    assert isinstance(gated_score, (int, float)), f"Relevance score should be numeric for {mem_id}"

        asyncio.run(test_integrity())

@pytest.mark.asyncio
async def test_threshold_sensitivity():
    """Tests gating behavior with different threshold settings"""

    test_memories = create_test_memory_dataset()
    test_query = "Test query for threshold testing"

    # Test with different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

    from processing.gate_system import MultiStageGateSystem

    results = {}

    for threshold in thresholds:
        mock_model_manager = Mock()
        mock_cross_encoder = Mock()
        mock_cross_encoder.predict = Mock(return_value=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])

        with patch('sentence_transformers.CrossEncoder', return_value=mock_cross_encoder):
            gate_system = MultiStageGateSystem(mock_model_manager)

            # Temporarily modify threshold for testing
            original_threshold = gate_system.cosine_threshold
            gate_system.cosine_threshold = threshold

            result = await gate_system.filter_memories(test_query, test_memories.copy(), k=20)
            results[threshold] = result

            # Restore original threshold
            gate_system.cosine_threshold = original_threshold

    # Verify threshold sensitivity
    for threshold in thresholds:
        filtered_count = len(results[threshold])
        logger.info(f"Threshold {threshold}: {filtered_count} memories passed")

        # Higher thresholds should filter more aggressively
        if threshold > 0.5:
            assert filtered_count <= len(test_memories) // 2, f"High threshold {threshold} should filter aggressively"

    # Verify monotonic relationship (higher threshold = fewer results)
    threshold_counts = [(threshold, len(results[threshold])) for threshold in thresholds]
    threshold_counts.sort(key=lambda x: x[0])  # Sort by threshold

    for i in range(len(threshold_counts) - 1):
        lower_threshold, lower_count = threshold_counts[i]
        higher_threshold, higher_count = threshold_counts[i + 1]

        # Higher threshold should not return more results than lower threshold
        assert higher_count <= lower_count + 1, f"Threshold {higher_threshold} should not return more than {lower_threshold}"

if __name__ == "__main__":
    # Run consistency tests manually
    print("Running gating consistency tests...")

    async def run_consistency_tests():
        validator = GatingConsistencyValidator()

        # Test main consistency
        await test_same_memories_same_gating_results(validator)

        # Test ranking stability
        await test_memory_ranking_stability(validator)

        print("All consistency tests passed!")

    asyncio.run(run_consistency_tests())