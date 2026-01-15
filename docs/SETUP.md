# Setup Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/zach8421/unlabeled-media-tagger.git
cd unlabeled-media-tagger
```

### 2. Create a Virtual Environment (Recommended)

```bash
# On Linux/Mac
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install the Package

```bash
# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### 4. Set Up Configuration

```bash
# Copy example configuration
cp examples/config.example.yaml config.yaml

# Or use .env for environment variables
cp .env.example .env
```

Edit the configuration file with your settings.

### 5. Google Drive Setup (Future)

When Google Drive integration is implemented:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Drive API
4. Create OAuth 2.0 credentials
5. Download the credentials JSON file
6. Save it as `credentials.json` in the project root

### 6. Model Setup (Future)

When model integration is implemented:

1. Download required model files
2. Place them in the `models/` directory
3. Update configuration with model paths

## Verification

Run the example script to verify installation:

```bash
python examples/example_pipeline.py
```

You should see the pipeline stages initialize successfully.

## Running Tests

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run with coverage
pytest --cov=unlabeled_media_tagger
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure the package is installed:
```bash
pip install -e .
```

### Python Version

Verify you're using Python 3.8+:
```bash
python --version
```

### Virtual Environment

Make sure your virtual environment is activated:
```bash
# You should see (venv) in your terminal prompt
```

## Next Steps

- Review the [Architecture Documentation](docs/ARCHITECTURE.md)
- Check out [Contributing Guidelines](CONTRIBUTING.md)
- Explore the [API Reference](docs/API.md)

## Getting Help

- Open an issue on GitHub
- Check existing documentation in `docs/`
- Review example code in `examples/`
