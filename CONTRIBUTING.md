# 🤝 Contributing to NeuroMind

Thank you for your interest in contributing to NeuroMind! This guide will help you get started with contributing to our EEG brain signal classification project.

## 🌟 Ways to Contribute

- 🐛 **Bug Reports**: Help us identify and fix issues
- ✨ **Feature Requests**: Suggest new functionality
- 🔧 **Code Contributions**: Implement features or fix bugs
- 📖 **Documentation**: Improve guides, tutorials, and API docs
- 🧪 **Testing**: Add tests and improve coverage
- 🎨 **UI/UX**: Enhance the Streamlit interface
- 🚀 **Performance**: Optimize model inference and deployment

## 🚀 Quick Start

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/neuromind-eeg-classifier.git
cd neuromind-eeg-classifier
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

## 🏗️ Development Workflow

### Code Standards

- **Python Style**: Follow PEP 8, use Black for formatting
- **Type Hints**: Add type annotations to all functions
- **Docstrings**: Use Google/NumPy style docstrings
- **Testing**: Write tests for new features (aim for 90%+ coverage)
- **Commits**: Use conventional commit messages

### Pre-commit Checks

Before submitting, ensure your code passes:

```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Run type checking
mypy src/

# Run tests
pytest tests/ -v --cov=src

# Check security
bandit -r src/
```

### File Structure Guidelines

```
src/
├── models/         # Model architectures and loading
├── data/          # Data processing and datasets
├── training/      # Training scripts and evaluation
├── utils/         # Utility functions and helpers
└── app.py         # Main Streamlit application

tests/             # Mirror src/ structure
docs/              # Documentation files
scripts/           # Automation and utility scripts
```

## 🐛 Bug Reports

When reporting bugs, please include:

- **Environment**: OS, Python version, dependency versions
- **Steps to Reproduce**: Clear step-by-step instructions
- **Expected vs Actual**: What should happen vs what actually happens
- **Error Messages**: Full stack traces if applicable
- **Screenshots**: For UI-related issues

### Bug Report Template

```markdown
**Environment:**
- OS: [e.g., Ubuntu 20.04, macOS 12.0, Windows 10]
- Python: [e.g., 3.9.7]
- NeuroMind version: [e.g., 1.0.0]

**Bug Description:**
[Clear description of the bug]

**Steps to Reproduce:**
1. [First step]
2. [Second step]
3. [See error]

**Expected Behavior:**
[What you expected to happen]

**Actual Behavior:**
[What actually happened]

**Error Messages:**
```
[Paste any error messages or stack traces here]
```

**Additional Context:**
[Any other relevant information]
```

## ✨ Feature Requests

For new features, please:

- **Check Existing Issues**: Search for similar requests
- **Describe Use Case**: Explain why this feature is needed
- **Propose Solution**: Suggest how it might work
- **Consider Alternatives**: What other approaches could work?

### Feature Request Template

```markdown
**Feature Summary:**
[Brief description of the feature]

**Problem/Use Case:**
[Describe the problem this feature would solve]

**Proposed Solution:**
[Detailed description of how this feature should work]

**Alternative Solutions:**
[Other ways this could be implemented]

**Additional Context:**
[Screenshots, mockups, examples, etc.]
```

## 🔧 Code Contributions

### Pull Request Process

1. **Create Issue First**: For significant changes, create an issue to discuss
2. **Follow Conventions**: Use our coding standards and commit format
3. **Add Tests**: Include tests for new functionality
4. **Update Docs**: Update documentation for API changes
5. **Small PRs**: Keep changes focused and reviewable

### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(models): add transformer architecture support
fix(preprocessing): handle edge case in bandpass filter
docs(readme): update installation instructions
test(gradcam): add comprehensive explainability tests
```

### Code Review Guidelines

**For Contributors:**
- Test your changes thoroughly
- Write clear, descriptive commit messages  
- Add comments for complex logic
- Update documentation as needed
- Be responsive to review feedback

**For Reviewers:**
- Be constructive and specific
- Focus on code quality, not personal preferences
- Suggest improvements with examples
- Approve when ready, request changes when needed

## 🧪 Testing Guidelines

### Test Types

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Benchmark critical paths

### Writing Tests

```python
import pytest
import numpy as np
from src.models.model import build_model

class TestModelBuilding:
    """Test model architecture creation."""
    
    def test_resnet18_architecture(self):
        """Test ResNet18 model creation."""
        model = build_model(arch="resnet18", pretrained=False)
        
        # Test model structure
        assert hasattr(model, 'fc')
        assert model.fc[-1].out_features == 3
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (1, 3)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py -v

# Run with markers
pytest tests/ -m "not slow" -v
```

## 📖 Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings, type hints, comments
2. **User Guides**: How-to tutorials and examples  
3. **API Reference**: Complete function/class documentation
4. **Architecture Docs**: System design and decisions

### Documentation Style

```python
def preprocess_eeg(raw: mne.io.Raw, 
                  low_freq: float = 4.0,
                  high_freq: float = 45.0) -> mne.io.Raw:
    """
    Preprocess EEG data with bandpass filtering and artifact removal.
    
    Args:
        raw: MNE Raw object containing EEG data
        low_freq: Lower frequency bound for bandpass filter (Hz)
        high_freq: Upper frequency bound for bandpass filter (Hz)
        
    Returns:
        Preprocessed MNE Raw object with filters applied
        
    Raises:
        ValueError: If frequency bounds are invalid
        
    Example:
        >>> raw = mne.io.read_raw_edf('data.edf')
        >>> clean_raw = preprocess_eeg(raw, low_freq=1.0, high_freq=40.0)
    """
```

## 🏆 Recognition

Contributors will be recognized in:

- **README Contributors Section**: All contributors listed
- **CHANGELOG**: Major contributions highlighted  
- **Release Notes**: Significant features credited
- **Documentation**: Author attribution where appropriate

## 🆘 Getting Help

- **GitHub Issues**: Ask questions about contributing
- **Discussions**: General questions and ideas
- **Email**: [neuromind.eeg@gmail.com](mailto:neuromind.eeg@gmail.com)

## 📋 Contribution Checklist

Before submitting a pull request:

- [ ] Fork the repository and create feature branch
- [ ] Code follows project style guidelines
- [ ] Added type hints and docstrings
- [ ] Tests added for new functionality
- [ ] All tests pass (`pytest tests/`)
- [ ] Documentation updated for changes
- [ ] Pre-commit hooks pass
- [ ] Commit messages follow convention
- [ ] Pull request description is clear

## 🎯 Good First Issues

Look for issues labeled `good-first-issue` or `help-wanted`:

- Documentation improvements
- Adding example notebooks
- Writing additional tests
- UI/UX enhancements in Streamlit app
- Performance optimizations
- Adding new visualization features

## 🌈 Code of Conduct

We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/):

- **Be Respectful**: Treat everyone with respect and kindness
- **Be Inclusive**: Welcome developers of all backgrounds and skill levels
- **Be Constructive**: Provide helpful feedback and suggestions
- **Be Professional**: Maintain a professional atmosphere

## 🚀 Advanced Contributions

### Adding New Model Architectures

1. Implement in `src/models/model.py`
2. Add comprehensive tests
3. Update documentation
4. Benchmark against existing models

### Extending Dataset Support

1. Add loader in `src/data/preprocessing.py`
2. Ensure compatibility with existing pipeline
3. Add validation and tests
4. Update documentation

### Performance Optimizations

1. Profile existing code to identify bottlenecks
2. Implement optimizations with benchmarks
3. Ensure accuracy is maintained
4. Add performance regression tests

---

**Thank you for contributing to NeuroMind! Together, we're advancing AI in healthcare and neuroscience.**

For questions about contributing, please open an issue or contact the maintainers.