"""Project-level reporting helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_markdown_table(df: pd.DataFrame, path: Path, title: str) -> None:
    """Persist a DataFrame as a Markdown section."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{df.to_markdown(index=False)}\n", encoding="utf-8")
