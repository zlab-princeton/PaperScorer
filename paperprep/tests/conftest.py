"""Pytest configuration for the paperprep test suite.

All tests live under ``tests/unit/``; they are self-contained and use
pytest's built-in ``tmp_path`` fixture for any I/O. No shared fixtures or
cache paths are needed at this level.

  pytest tests/unit/                 — run everything
  pytest tests/unit/paperprep/       — only the paperprep CLI/manifest/state tests
"""
