# Contributing to Solari

We welcome contributions. Here's how to get started.

## Getting Started

```bash
git clone https://github.com/SolariResearch/solari.git
cd solari
pip install -e ".[all]"
python -m pytest tests/
```

## What We're Looking For

- New ingest source types (RSS feeds, Notion, Confluence, etc.)
- Performance improvements to FAISS indexing
- Additional workspace processors
- Documentation and examples
- Bug fixes with tests

## Pull Requests

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`python -m pytest`)
5. Submit a PR with a clear description

## Code Style

- Python 3.10+
- Type hints on public functions
- Docstrings on modules and classes
- Keep dependencies minimal

## License

By contributing, you agree that your contributions will be licensed under the AGPL-3.0 license.
