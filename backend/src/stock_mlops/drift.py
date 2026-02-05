from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

import pandas as pd

from .logging_config import setup_logging

logger = setup_logging(__name__)


@dataclass
class DriftResult:
    success: bool
    summary: Dict[str, Any]
    report_path: Optional[str] = None


def run_evidently_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    report_dir: str = "artifacts/drift",
    report_name: str = "data_drift_report.html",
) -> DriftResult:
    """Run an Evidently data drift report and save it as HTML.

    This is intentionally lightweight: we use a preset to compute drift metrics.
    """
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(report_dir) / report_name)

    try:
        # Evidently v0.4+ style
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        report.save_html(out_path)

        # Extract a minimal summary if possible
        summary = {"report": "saved", "path": out_path}
        logger.info("Evidently drift report saved: %s", out_path)
        return DriftResult(success=True, summary=summary, report_path=out_path)

    except Exception as e:
        logger.warning("Evidently drift report failed: %s", e)
        return DriftResult(success=False, summary={"error": str(e)}, report_path=None)
