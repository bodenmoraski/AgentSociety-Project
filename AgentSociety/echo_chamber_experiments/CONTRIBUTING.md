# 🤝 Contributing to Dynamic Belief Evolution Framework

Thank you for your interest in contributing to the Dynamic Belief Evolution Framework! This document provides guidelines and information for contributors.

## 🎯 How to Contribute

### Types of Contributions Welcome

- 🐛 **Bug Reports** - Help us identify and fix issues
- 💡 **Feature Requests** - Suggest new capabilities and improvements  
- 📝 **Documentation** - Improve guides, examples, and API docs
- 🧪 **Test Coverage** - Add tests for better reliability
- 🎨 **Visualizations** - Create new plot types and dashboards
- 🔬 **Research Applications** - Share use cases and domain extensions
- ⚡ **Performance** - Optimize algorithms and memory usage

## 🚀 Getting Started

### 1. Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/echo-chamber-experiments.git
cd echo-chamber-experiments

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development tools

# Verify setup
python tests/test_installation.py
```

### 2. Development Tools

```bash
# Code formatting
black .
isort .

# Linting
flake8 .
pylint core/ experiments/

# Type checking  
mypy core/ experiments/

# Testing
pytest tests/ -v --cov=core --cov=experiments
```

### 3. Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## 📋 Development Guidelines

### Code Style

- **PEP 8** compliance with 88-character line length
- **Type hints** for all function signatures
- **Docstrings** following Google style
- **Clear variable names** and meaningful comments

#### Example Function

```python
def analyze_belief_trajectory(
    trajectory: List[float], 
    method: str = 'polynomial',
    degree: int = 3
) -> TrajectoryAnalysis:
    """
    Analyze agent belief trajectory using specified mathematical method.
    
    Args:
        trajectory: List of belief values over time [-1, 1]
        method: Analysis method ('polynomial', 'spline', 'fourier')
        degree: Polynomial degree or spline order (default: 3)
        
    Returns:
        TrajectoryAnalysis object with fitted parameters and metrics
        
    Raises:
        ValueError: If trajectory is empty or method is invalid
        
    Example:
        >>> trajectory = [0.1, 0.3, 0.5, 0.2]
        >>> analysis = analyze_belief_trajectory(trajectory, 'polynomial', 2)
        >>> print(analysis.r_squared)
        0.923
    """
    if not trajectory:
        raise ValueError("Trajectory cannot be empty")
        
    # Implementation here...
    return TrajectoryAnalysis(...)
```

### Testing Requirements

- **Unit tests** for all new functions
- **Integration tests** for new features
- **Regression tests** for bug fixes
- **Performance tests** for optimization changes

#### Test Structure

```python
import pytest
import numpy as np
from core.dynamic_parameters import DynamicBeliefParameters

class TestDynamicBeliefParameters:
    """Test suite for dynamic belief parameter system."""
    
    def test_parameter_interpolation_linear(self):
        """Test linear interpolation between keyframes."""
        # Arrange
        params = create_test_parameters()
        
        # Act
        result = params.get_parameters_at_time(5)
        
        # Assert
        assert 0.0 <= result.polarization_strength <= 1.0
        assert result.distribution_type is not None
        
    def test_parameter_validation_bounds(self):
        """Test parameter bound validation."""
        with pytest.raises(ValueError, match="Polarization strength"):
            create_invalid_parameters(polarization_strength=1.5)
            
    @pytest.mark.parametrize("method", ['linear', 'cubic_spline', 'sigmoid'])
    def test_interpolation_methods(self, method):
        """Test all interpolation methods work correctly."""
        params = create_test_parameters(interpolation=method)
        result = params.get_parameters_at_time(5)
        assert result is not None
```

### Documentation Standards

- **Comprehensive docstrings** for all public functions
- **Type hints** for better IDE support
- **Usage examples** in docstrings
- **Mathematical explanations** for complex algorithms

## 🐛 Reporting Bugs

### Bug Report Template

```markdown
## Bug Description
Brief description of the issue

## Reproduction Steps
1. Step one
2. Step two  
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., macOS 12.0, Ubuntu 20.04, Windows 11]
- Python: [e.g., 3.9.7]
- Framework Version: [e.g., 2.0.0]

## Additional Context
- Error messages (full traceback)
- Configuration files
- Sample data (if applicable)
```

### Before Submitting

1. **Search existing issues** - Check if already reported
2. **Run sanity checks** - `python tests/test_sanity_checks.py`
3. **Minimal reproduction** - Provide smallest possible example
4. **Environment details** - Include system information

## 💡 Feature Requests

### Feature Request Template

```markdown
## Feature Description
Clear description of the proposed feature

## Use Case
Why is this feature needed? What problem does it solve?

## Proposed Implementation
How might this work? (technical details welcome)

## Alternative Solutions
Other approaches you've considered

## Additional Context
Mockups, research papers, related work
```

### Design Principles

- **Research-First** - Features should enable novel research
- **Mathematical Rigor** - Algorithms should be theoretically sound
- **Performance Conscious** - Consider scalability implications
- **User-Friendly** - APIs should be intuitive and well-documented

## 🔄 Pull Request Process

### 1. Before Starting

- **Open an issue** to discuss major changes
- **Check roadmap** to avoid conflicts with planned work
- **Fork the repository** and create a feature branch

### 2. Development Process

```bash
# Create feature branch
git checkout -b feature/amazing-new-feature

# Make changes
# ... develop and test ...

# Run full test suite
python tests/test_sanity_checks.py
pytest tests/ -v --cov=core --cov=experiments

# Check code quality
black .
flake8 .
mypy core/ experiments/

# Commit changes
git add .
git commit -m "feat: add amazing new feature

- Detailed description of changes
- List of new capabilities
- Breaking changes (if any)"

# Push to fork
git push origin feature/amazing-new-feature
```

### 3. Pull Request Checklist

- [ ] **Tests pass** - All existing and new tests
- [ ] **Documentation updated** - READMEs, docstrings, guides
- [ ] **Code quality** - Linting, formatting, type hints
- [ ] **Performance** - No significant regressions
- [ ] **Backward compatibility** - Or clear migration path
- [ ] **Examples updated** - If API changes affect demos

### 4. Review Process

1. **Automated checks** - CI/CD pipeline validation
2. **Code review** - At least one maintainer approval
3. **Testing** - Manual verification if needed
4. **Documentation review** - Ensure clarity and completeness
5. **Merge** - Squash and merge with descriptive commit message

## 🧪 Testing Guidelines

### Test Categories

#### Unit Tests
```python
# Test individual functions/methods
def test_belief_distribution_generation():
    """Test continuous belief distribution generation."""
    params = BeliefDistributionParams(
        distribution_type=DistributionType.BIMODAL,
        polarization_strength=0.8
    )
    generator = ContinuousBeliefGenerator(params)
    beliefs = generator.generate_beliefs(100)
    
    assert len(beliefs) == 100
    assert all(-1 <= b <= 1 for b in beliefs)
```

#### Integration Tests
```python
# Test component interactions
def test_experiment_pipeline():
    """Test complete experiment execution pipeline."""
    config = create_test_config()
    experiment = DynamicEvolutionExperiment(config)
    results = experiment.run_full_experiment()
    
    assert results.polarization_over_time is not None
    assert len(results.belief_trajectories) == config.num_agents
```

#### Performance Tests
```python
# Test execution speed and memory usage
def test_large_experiment_performance():
    """Test performance with large agent populations."""
    config = DynamicEvolutionConfig(num_agents=1000, num_rounds=50)
    
    start_time = time.time()
    experiment = DynamicEvolutionExperiment(config)
    results = experiment.run_full_experiment()
    duration = time.time() - start_time
    
    assert duration < 60.0  # Should complete within 1 minute
    assert results is not None
```

### Test Data

- **Fixtures** - Use pytest fixtures for reusable test data
- **Deterministic** - Set random seeds for reproducible tests
- **Edge Cases** - Test boundary conditions and error cases
- **Realistic** - Use parameters similar to actual research

## 📝 Documentation Guidelines

### Documentation Types

#### API Documentation
- **Docstrings** for all public functions
- **Type hints** for better tooling support
- **Examples** showing typical usage
- **Parameter descriptions** with valid ranges

#### User Guides
- **Step-by-step tutorials** for common tasks
- **Best practices** and recommendations
- **Troubleshooting** common issues
- **Research applications** and case studies

#### Developer Documentation
- **Architecture overviews** explaining design decisions
- **Algorithm explanations** with mathematical background
- **Performance considerations** and optimization tips
- **Extension points** for custom functionality

### Documentation Standards

- **Clear and concise** writing
- **Code examples** that actually work
- **Visual aids** where helpful (diagrams, plots)
- **Regular updates** to match code changes

## 🌟 Recognition

### Contributors

All contributors will be recognized in:
- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **Documentation** author attributions
- **Research citations** for algorithmic contributions

### Types of Recognition

- **Code contributions** - Bug fixes, features, optimizations
- **Documentation** - Guides, examples, API docs
- **Research** - Novel algorithms, validation studies
- **Community** - Issue triage, user support, advocacy

## 📞 Getting Help

### Communication Channels

- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - General questions and ideas
- **Code Reviews** - Feedback on pull requests
- **Documentation** - Comprehensive guides and examples

### Response Times

- **Bug reports** - Within 48 hours
- **Feature requests** - Within 1 week  
- **Pull requests** - Within 1 week
- **General questions** - Within 72 hours

*Note: Response times may vary based on complexity and maintainer availability*

## 📜 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**🙏 Thank you for contributing to the Dynamic Belief Evolution Framework!**

Your contributions help advance computational social science research and enable better understanding of belief dynamics in crisis scenarios.

**Questions?** Feel free to open an issue or start a discussion - we're here to help!