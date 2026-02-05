from __future__ import annotations

import os
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from .logging_config import setup_logging
from .settings import settings

logger = setup_logging(__name__)

def push_rmse(rmse: float, job: str = "stock-forecast") -> None:
    """Push RMSE to Prometheus Pushgateway for Grafana dashboards/alerts."""
    registry = CollectorRegistry()
    g = Gauge("model_rmse", "RMSE of latest evaluation", registry=registry)
    g.set(rmse)

    url = os.getenv("PUSHGATEWAY_URL", settings.pushgateway_url)
    try:
        push_to_gateway(url, job=job, registry=registry)
        logger.info("Pushed RMSE=%.6f to Pushgateway (%s).", rmse, url)
    except Exception as e:
        logger.warning("Failed to push metrics to Pushgateway (%s): %s", url, e)
