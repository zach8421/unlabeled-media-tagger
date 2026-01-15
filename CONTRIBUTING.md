# Contributing to unlabeled-media-tagger

Thank you for your interest in contributing to unlabeled-media-tagger! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/unlabeled-media-tagger.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `pytest`
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## 🛠️ Development Setup

```bash
# Install the package in development mode
pip install -e ".[dev]"

# Install development dependencies
pip install -r requirements-dev.txt
```

## 🧪 Testing

- Write tests for new features
- Ensure all tests pass before submitting a PR
- Run tests with: `pytest`
- Check coverage with: `pytest --cov=unlabeled_media_tagger`

## 📝 Code Style

- Follow PEP 8 style guidelines
- Use Black for code formatting: `black src/ tests/`
- Use flake8 for linting: `flake8 src/ tests/`
- Add type hints where appropriate
- Write docstrings for all public functions and classes

## 🎯 Areas for Contribution

### High Priority

1. **Google Drive API Integration** (fetch stage)
   - OAuth2 authentication
   - File querying and downloading
   - Metadata reading/writing

2. **Frame Extraction** (extract stage)
   - Video frame extraction using OpenCV
   - EXIF metadata reading
   - Timestamp extraction

3. **Computer Vision Models** (detect stage)
   - Face detection model integration
   - Face recognition/embedding generation
   - Object detection model integration

4. **Face Clustering** (compare stage)
   - Embedding comparison algorithms
   - Clustering implementation
   - Face database management

5. **Metadata Enrichment** (enrich stage)
   - Local file metadata writing
   - Google Drive metadata updates
   - Tag generation

### Other Contributions

- Documentation improvements
- Bug fixes
- Performance optimizations
- Additional test coverage
- Example scripts and tutorials

## 📋 Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Include tests for new functionality
- Update documentation as needed
- Ensure CI checks pass

## 🐛 Reporting Issues

When reporting issues, please include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant error messages or logs

## 💬 Questions?

Feel free to open an issue for questions or discussions about the project.

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.
