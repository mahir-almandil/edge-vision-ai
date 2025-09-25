# Contributing to Edge Vision AI

Thank you for your interest in contributing to Edge Vision AI! We welcome contributions from the community and are grateful for any help you can provide.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please treat all contributors with respect and professionalism.

## How to Contribute

### Reporting Issues

1. Check if the issue already exists in the [issue tracker](https://github.com/yourusername/edge-vision-ai/issues)
2. Create a new issue with a clear title and description
3. Include:
   - Steps to reproduce the issue
   - Expected behavior
   - Actual behavior
   - System information (OS, Python version, hardware)
   - Error messages and logs

### Suggesting Features

1. Check if the feature has already been requested
2. Open a new issue with the `enhancement` label
3. Describe the feature and its use case
4. Explain why it would be valuable to the project

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Write or update tests as needed
5. Ensure all tests pass (`pytest tests/`)
6. Run code formatting (`black src/`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to your branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Local Development

```bash
# Clone your fork
git clone https://github.com/yourusername/edge-vision-ai.git
cd edge-vision-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/
```

### Docker Development

```bash
# Build development image
docker-compose --profile dev up

# Access the development container
docker exec -it edge-vision-dev bash
```

## Coding Standards

### Python Style Guide

- Follow PEP 8
- Use Black for code formatting
- Maximum line length: 120 characters
- Use type hints where appropriate
- Write docstrings for all public functions and classes

### Commit Messages

- Use clear and descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Keep the first line under 50 characters
- Add detailed description after a blank line if needed

Example:
```
Add PPE detection feature

- Implement helmet and vest detection
- Add color-based detection algorithm
- Update documentation with new feature
```

### Testing

- Write tests for new features
- Maintain or improve code coverage
- Use pytest for testing
- Place tests in the `tests/` directory

## Documentation

- Update README.md if adding new features
- Add docstrings to new functions and classes
- Update API documentation if changing endpoints
- Include examples for new functionality

## Release Process

1. Update version number in `setup.py`
2. Update CHANGELOG.md
3. Create a pull request to `main` branch
4. After merge, create a release tag
5. GitHub Actions will automatically build and deploy

## Getting Help

- Join our [Discord server](https://discord.gg/edgevisionai)
- Ask questions in [GitHub Discussions](https://github.com/yourusername/edge-vision-ai/discussions)
- Check the [documentation](docs/)
- Email the maintainers at support@edgevisionai.com

## Recognition

Contributors will be recognized in:
- The project README
- Release notes
- Our website's contributors page

## License

By contributing to Edge Vision AI, you agree that your contributions will be licensed under the MIT License.