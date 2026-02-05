from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import pandas as pd
import great_expectations as ge

from .logging_config import setup_logging

logger = setup_logging(__name__)


@dataclass
class ValidationResult:
    success: bool
    summary: Dict[str, Any]


def validate_yfinance_frame(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
) -> ValidationResult:
    """Lightweight Great Expectations validation (no GE project scaffolding needed)."""
    required_columns = required_columns or ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    gdf = ge.from_pandas(df.reset_index(drop=False))

    # Basic schema checks
    for col in required_columns:
        gdf.expect_column_to_exist(col)

    # Non-null & basic value sanity
    for col in ["Open", "High", "Low", "Close", "Adj Close"]:
        gdf.expect_column_values_to_not_be_null(col)
        gdf.expect_column_values_to_be_between(col, min_value=0, mostly=0.999)

    gdf.expect_column_values_to_not_be_null("Volume")
    gdf.expect_column_values_to_be_between("Volume", min_value=0, mostly=0.999)

    res = gdf.validate()
    summary = {
        "success": bool(res.get("success")),
        "statistics": res.get("statistics", {}),
        "failed_expectations": [
            {
                "expectation_type": r.get("expectation_config", {}).get("expectation_type"),
                "kwargs": r.get("expectation_config", {}).get("kwargs"),
                "success": r.get("success"),
            }
            for r in res.get("results", [])
            if not r.get("success", False)
        ][:25],
    }

    if not summary["success"]:
        logger.warning("Data validation failed: %s", summary["failed_expectations"][:3])
    else:
        logger.info("Data validation passed.")

    return ValidationResult(success=summary["success"], summary=summary)
