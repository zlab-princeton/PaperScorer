"""PaperLens CLI: setup-model, setup-data, serve.

A console_script entrypoint that bundles the three operator-side commands
the deployment needs in one place. The heavy lifting (vLLM scoring, LF
tokenization, sharegpt reconstruction) lives in the underlying scripts and
this package's siblings -- the CLI just wires them together.
"""

__version__ = "0.1.0"
