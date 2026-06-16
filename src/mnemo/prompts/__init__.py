"""Mnemo built-in prompt resources.

Provides the path to ``builtin.yaml`` so ``PromptManager``
can load prompts from the package data directory at runtime.
"""

from pathlib import Path

_BUILTIN_PROMPTS_PATH = Path(__file__).parent / "builtin.toml"
