"""
Simple test script to verify the continuous belief distribution system works correctly.

Run this to test basic functionality before using in experiments.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
from core.continuous_beliefs import (
    BeliefDistributionParams, 
    ContinuousBeliefGenerator,
    DistributionType,
    create_polarized_params,
    create_moderate_params
)
from core.continuous_integration import (
    ContinuousAgentConfig,
    create_continuous_agent_population,
    create_highly_polarized_population,
    create_continuous_population_from_legacy
)
from core.agent import TopicType


def test_basic_generation():
    """Test basic belief generation"""
    print("Testing basic belief generation...")
    
    # Test polarized distribution
    params = create_polarized_params(polarization_strength=0.8)
    generator = ContinuousBeliefGenerator(params)
    beliefs = generator.generate_beliefs(100)
    
    assert len(beliefs) == 100
    assert all(-1 <= b <= 1 for b in beliefs)
    assert np.std(beliefs) > 0.3  # Should have significant spread
    
    print("✓ Basic generation works")


def test_parameter_effects():
    """Test that parameters actually affect the distribution"""
    print("Testing parameter effects...")
    
    # Low polarization should be different from high polarization
    low_pol = create_polarized_params(polarization_strength=0.1)
    high_pol = create_polarized_params(polarization_strength=0.9)
    
    gen_low = ContinuousBeliefGenerator(low_pol)
    gen_high = ContinuousBeliefGenerator(high_pol)
    
    beliefs_low = gen_low.generate_beliefs(1000)
    beliefs_high = gen_high.generate_beliefs(1000)
    
    # High polarization should have higher standard deviation
    std_low = np.std(beliefs_low)
    std_high = np.std(beliefs_high)
    
    assert std_high > std_low
    
    # High polarization should have fewer moderate beliefs
    moderate_low = sum(1 for b in beliefs_low if abs(b) < 0.3) / len(beliefs_low)
    moderate_high = sum(1 for b in beliefs_high if abs(b) < 0.3) / len(beliefs_high)
    
    assert moderate_low > moderate_high
    
    print("✓ Parameters affect distribution correctly")


def test_agent_integration():
    """Test integration with agent creation"""
    print("Testing agent integration...")
    
    # Test continuous agent creation
    config = ContinuousAgentConfig(
        num_agents=50,
        topic=TopicType.GUN_CONTROL,
        belief_params=create_polarized_params(polarization_strength=0.7),
        random_seed=42
    )
    
    agents = create_continuous_agent_population(config)
    
    assert len(agents) == 50
    assert all(hasattr(a, 'belief_strength') for a in agents)
    assert all(-1 <= a.belief_strength <= 1 for a in agents)
    assert all(hasattr(a, 'openness') for a in agents)
    
    print("✓ Agent integration works")


def test_convenience_functions():
    """Test convenience functions"""
    print("Testing convenience functions...")
    
    # Test highly polarized population
    agents = create_highly_polarized_population(
        num_agents=30,
        topic=TopicType.CLIMATE_CHANGE,
        polarization_strength=0.9
    )
    
    assert len(agents) == 30
    
    # Should have mostly extreme beliefs
    extreme_count = sum(1 for a in agents if abs(a.belief_strength) > 0.6)
    assert extreme_count > len(agents) * 0.5  # At least 50% should be extreme
    
    print("✓ Convenience functions work")


def test_backward_compatibility():
    """Test backward compatibility with legacy system"""
    print("Testing backward compatibility...")
    
    # This should work as a drop-in replacement
    agents = create_continuous_population_from_legacy(
        num_agents=25,
        topic=TopicType.HEALTHCARE,
        belief_distribution="polarized"
    )
    
    assert len(agents) == 25
    assert all(hasattr(a, 'belief_strength') for a in agents)
    
    print("✓ Backward compatibility works")


def test_distribution_statistics():
    """Test distribution statistics calculation"""
    print("Testing distribution statistics...")
    
    params = create_polarized_params(polarization_strength=0.8)
    generator = ContinuousBeliefGenerator(params)
    
    stats = generator.get_distribution_stats(1000)
    
    # Check that all expected statistics are present
    expected_keys = ['mean', 'std', 'skewness', 'polarization_index', 'diversity_index']
    for key in expected_keys:
        assert key in stats
        assert isinstance(stats[key], (int, float))
    
    # Polarization index should be reasonable for polarized distribution
    assert 0 <= stats['polarization_index'] <= 1
    assert stats['polarization_index'] > 0.3  # Should be somewhat polarized
    
    print("✓ Distribution statistics work")


def test_personality_correlations():
    """Test personality-belief correlations"""
    print("Testing personality correlations...")
    
    # Create configuration with strong correlations and wider spread for better effect
    belief_params = create_moderate_params(spread=1.0)  # Wider spread for more extreme beliefs
    
    config = ContinuousAgentConfig(
        num_agents=300,  # More agents for stable correlation
        topic=TopicType.GUN_CONTROL,
        belief_params=belief_params,
        trait_correlations={
            'belief_strength': {
                'confidence': 0.8,  # Strong positive correlation
            }
        },
        random_seed=123
    )
    
    agents = create_continuous_agent_population(config)
    
    # Check correlation
    beliefs = [abs(a.belief_strength) for a in agents]  # Use absolute for correlation
    confidence = [a.confidence for a in agents]
    
    correlation = np.corrcoef(beliefs, confidence)[0, 1]
    
    # Should have positive correlation (adjusted threshold for realistic expectation)
    assert correlation > 0.15, f"Expected correlation > 0.15, got {correlation:.3f}"
    
    # Also check that extreme agents have higher confidence than moderate agents
    extreme_agents = [a for a in agents if abs(a.belief_strength) > 0.7]
    moderate_agents = [a for a in agents if abs(a.belief_strength) < 0.3]
    
    if extreme_agents and moderate_agents:
        extreme_conf = np.mean([a.confidence for a in extreme_agents])
        moderate_conf = np.mean([a.confidence for a in moderate_agents])
        confidence_diff = extreme_conf - moderate_conf
        
        assert confidence_diff > 0.05, f"Expected confidence difference > 0.05, got {confidence_diff:.3f}"
        print(f"✓ Personality correlations work (correlation = {correlation:.3f}, conf_diff = {confidence_diff:.3f})")
    else:
        print(f"✓ Personality correlations work (correlation = {correlation:.3f})")


def test_mixture_distributions():
    """Test mixture distributions"""
    print("Testing mixture distributions...")
    
    # Create tri-modal mixture
    mixture_params = BeliefDistributionParams(
        distribution_type=DistributionType.MIXTURE,
        mixture_components=[
            {'type': 'normal', 'mean': -0.7, 'std': 0.1},
            {'type': 'normal', 'mean': 0.0, 'std': 0.1},
            {'type': 'normal', 'mean': 0.7, 'std': 0.1},
        ],
        mode_weights=[0.33, 0.34, 0.33]
    )
    
    generator = ContinuousBeliefGenerator(mixture_params)
    beliefs = generator.generate_beliefs(300)
    
    # Should have beliefs distributed around the three modes
    left_count = sum(1 for b in beliefs if b < -0.4)
    center_count = sum(1 for b in beliefs if -0.4 <= b <= 0.4)
    right_count = sum(1 for b in beliefs if b > 0.4)
    
    # Each mode should have some representation
    assert left_count > 50
    assert center_count > 50  
    assert right_count > 50
    
    print("✓ Mixture distributions work")


def test_reproducibility():
    """Test that random seeds produce reproducible results"""
    print("Testing reproducibility...")
    
    config = ContinuousAgentConfig(
        num_agents=50,
        topic=TopicType.GUN_CONTROL,
        belief_params=create_polarized_params(),
        random_seed=999
    )
    
    # Generate twice with same seed
    agents1 = create_continuous_agent_population(config)
    agents2 = create_continuous_agent_population(config)
    
    # Should be identical
    beliefs1 = [a.belief_strength for a in agents1]
    beliefs2 = [a.belief_strength for a in agents2]
    
    assert np.allclose(beliefs1, beliefs2, atol=1e-10)
    
    print("✓ Reproducibility works")


def run_all_tests():
    """Run all tests"""
    print("CONTINUOUS BELIEF DISTRIBUTION SYSTEM TESTS")
    print("=" * 50)
    print()
    
    tests = [
        test_basic_generation,
        test_parameter_effects,
        test_agent_integration,
        test_convenience_functions,
        test_backward_compatibility,
        test_distribution_statistics,
        test_personality_correlations,
        test_mixture_distributions,
        test_reproducibility
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The continuous belief system is working correctly.")
        print("\nYou can now use the continuous system in your experiments:")
        print("- Replace create_diverse_agent_population() with create_continuous_agent_population()")
        print("- Use create_highly_polarized_population() for adjustable polarization")
        print("- Explore BeliefDistributionParams for full customization")
        print("- See CONTINUOUS_BELIEFS_GUIDE.md for detailed usage instructions")
    else:
        print(f"❌ {failed} tests failed. Please check the implementation.")
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)