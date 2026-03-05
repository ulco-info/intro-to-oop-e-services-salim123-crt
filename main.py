

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.reporting.health_reporter import get_health_status, HealthReportGenerator
import os

app = FastAPI(
    title="PricePulse AI",
    description="AI-powered market price recommendation system",
    version=os.getenv("APP_VERSION", "1.0.0"),
)


@app.get("/health", tags=["DevOps"])
async def health_check():
    """Lightweight liveness probe used by Docker HEALTHCHECK."""
    return JSONResponse(content=get_health_status())


@app.get("/health/full", tags=["DevOps"])
async def full_health_report():
    """
    Triggers a full health report generation and returns the JSON result.
    This hits all external services so it may take a few seconds.
    """
    generator = HealthReportGenerator(
        output_dir=os.getenv("REPORT_OUTPUT_DIR", "/app/reports"),
        report_format="json",
    )
    report = generator.generate()
    return JSONResponse(content=report.to_dict())


@app.get("/metrics", tags=["DevOps"])
async def scraper_metrics():
    """Returns current scraper KPIs (stub — hooked by Data Engineers)."""
    from app.reporting.health_reporter import MetricsCollector
    from dataclasses import asdict
    collector = MetricsCollector()
    metrics = collector.collect()
    return {"scrapers": [asdict(m) for m in metrics]}
