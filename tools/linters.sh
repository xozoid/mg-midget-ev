#!/bin/sh
# Run every linter
ruff check .
codespell . -S node_modules
basedpyright # No dot cause it gets included files from pyproject.toml
