"""Robust loading and cleaning utilities for the tourism demand project."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.city_mapping import CITY_MAPPING


LOGGER = logging.getLogger(__name__)


def configure_logging(log_path: Path) -> None:
    """Configure file and console logging once for the pipeline."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def detect_separator(path: Path) -> str:
    """Detect a CSV separator from the first non-empty line."""
    sample = path.read_text(encoding="utf-8", errors="replace").splitlines()
    first_line = next((line for line in sample if line.strip()), "")
    return ";" if first_line.count(";") > first_line.count(",") else ","


def load_table(path: Path) -> pd.DataFrame:
    """Load CSV, Excel or JSON while logging shape, dtypes and columns."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(path, sep=detect_separator(path), encoding="utf-8")
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        elif suffix == ".json":
            with path.open(encoding="utf-8") as handle:
                payload: Any = json.load(handle)
            df = pd.json_normalize(payload)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
    except Exception:
        LOGGER.exception("Cannot load %s", path)
        raise

    LOGGER.info("Loaded %s | shape=%s | columns=%s", path.name, df.shape, list(df.columns))
    LOGGER.info("Dtypes for %s:\n%s", path.name, df.dtypes)
    return df


def standardize_text(value: object) -> str | None:
    """Clean business keys without inventing missing information."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    return " ".join(text.split()) if text else None


def normalize_country(value: object) -> str | None:
    """Harmonize country labels while preserving unknown values as missing."""
    text = standardize_text(value)
    if not text:
        return None
    if text.upper() == "USA":
        return "USA"
    return text.title()


def normalize_destination(value: object) -> str | None:
    """Harmonize destination labels such as City_3 and city_3."""
    text = standardize_text(value)
    if not text:
        return None
    if "_" in text:
        prefix, suffix = text.split("_", 1)
        return f"{prefix.title()}_{suffix}"
    return text.title()


def apply_destination_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Replace anonymized destination codes while preserving lineage."""
    if "destination" not in df.columns or "country" not in df.columns:
        return df
    mapped = df.copy()
    mapped["destination_original"] = mapped["destination"]
    mapped["destination"] = mapped.apply(
        lambda row: CITY_MAPPING.get(
            (row["country"], row["destination_original"]),
            row["destination_original"],
        ),
        axis=1,
    )
    return mapped


def numeric_budget(value: object) -> float | None:
    """Parse campaign budgets and keep non-numeric values as missing."""
    if pd.isna(value):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        LOGGER.warning("Non numeric campaign budget encountered: %s", value)
        return None


def clean_destinations(df: pd.DataFrame) -> pd.DataFrame:
    """Clean destination master data with conservative quality rules."""
    cleaned = df.copy()
    cleaned["country"] = cleaned["country"].map(normalize_country)
    cleaned["destination"] = cleaned["destination"].map(normalize_destination)
    cleaned = apply_destination_mapping(cleaned)
    for col in ["attractiveness", "cost", "rating", "visitors"]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned = cleaned.drop_duplicates(subset=["country", "destination"], keep="first")
    return cleaned


def clean_market(df: pd.DataFrame) -> pd.DataFrame:
    """Clean monthly country demand signals."""
    cleaned = df.copy()
    cleaned["country"] = cleaned["country"].map(normalize_country)
    cleaned["month"] = pd.to_datetime(cleaned["month"], errors="coerce")
    cleaned["demand_index"] = pd.to_numeric(cleaned["demand_index"], errors="coerce")
    return cleaned.dropna(subset=["country", "month", "demand_index"])


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Clean social review records and keep sentiment as evidence, not truth."""
    cleaned = df.copy()
    cleaned["country"] = cleaned["country"].map(normalize_country)
    cleaned["destination"] = cleaned["destination"].map(normalize_destination)
    cleaned = apply_destination_mapping(cleaned)
    cleaned["sentiment"] = cleaned["sentiment"].map(lambda x: standardize_text(x).lower() if standardize_text(x) else None)
    cleaned["score"] = pd.to_numeric(cleaned["score"], errors="coerce")
    return cleaned


def clean_external(df: pd.DataFrame) -> pd.DataFrame:
    """Clean external factors with ordered categorical business mappings."""
    cleaned = df.copy()
    cleaned["country"] = cleaned["country"].map(normalize_country)
    cleaned["destination"] = cleaned["destination"].map(normalize_destination)
    cleaned = apply_destination_mapping(cleaned)
    cleaned["flight_price"] = cleaned["flight_price"].map(lambda x: standardize_text(x).lower() if standardize_text(x) else None)
    cleaned["weather"] = cleaned["weather"].map(lambda x: standardize_text(x).lower() if standardize_text(x) else None)
    return cleaned


def clean_campaigns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean campaign history and flag contradictions for governance review."""
    cleaned = df.copy()
    cleaned["country"] = cleaned["country"].map(normalize_country)
    cleaned["destination"] = cleaned["destination"].map(normalize_destination)
    cleaned = apply_destination_mapping(cleaned)
    cleaned["campaign_budget"] = cleaned["campaign_budget"].map(numeric_budget)
    cleaned["conversion_rate"] = pd.to_numeric(cleaned["conversion_rate"], errors="coerce")
    cleaned["roi"] = pd.to_numeric(cleaned["roi"], errors="coerce")
    cleaned["status"] = cleaned["status"].map(lambda x: standardize_text(x).upper() if standardize_text(x) else None)
    cleaned["campaign_contradiction"] = (
        ((cleaned["status"] == "SUCCESS") & (cleaned["roi"] <= 0))
        | ((cleaned["status"] == "FAIL") & (cleaned["roi"] > 0))
    )
    return cleaned


def quality_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Return column-level data quality indicators."""
    rows = []
    for column in df.columns:
        rows.append(
            {
                "dataset": dataset_name,
                "column": column,
                "dtype": str(df[column].dtype),
                "rows": len(df),
                "missing": int(df[column].isna().sum()),
                "missing_rate": float(df[column].isna().mean()),
                "unique_values": int(df[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)
