

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pricepulse.health_reporter")


class ReportGenerationError(Exception):
    """Raised when the report cannot be generated or persisted."""


class WebhookDeliveryError(Exception):
    """Raised when the alert webhook call fails."""


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes (DTOs)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ServiceStatus:
    """Health status snapshot for a single service / dependency."""
    name: str
    status: str                    # "healthy" | "degraded" | "down" | "unknown"
    latency_ms: Optional[float]    # Round-trip time in milliseconds
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def is_healthy(self) -> bool:
        return self.status == "healthy"


@dataclass
class ScraperMetrics:
    """Aggregate statistics collected from market scrapers."""
    provider: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    listings_fetched: int = 0
    last_error: Optional[str] = None

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def success_rate(self) -> float:
        return 1.0 - self.error_rate


@dataclass
class HealthReport:
    """
    Top-level health report for the entire PricePulse system.
    Aggregates all service statuses and scraper metrics.
    """
    report_id: str
    generated_at: str
    app_version: str
    environment: str
    overall_status: str                        # "healthy" | "degraded" | "critical"
    services: List[ServiceStatus] = field(default_factory=list)
    scraper_metrics: List[ScraperMetrics] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Health Checkers
# ─────────────────────────────────────────────────────────────────────────────

class BaseHealthChecker:
    """Abstract-style base for all health checkers."""
    name: str = "base"

    def check(self) -> ServiceStatus:
        raise NotImplementedError


class APIHealthChecker(BaseHealthChecker):
    """Checks the FastAPI /health endpoint."""
    name = "pricepulse-api"

    def __init__(self, base_url: str = "http://localhost:8000"):
        self._base_url = base_url.rstrip("/")

    def check(self) -> ServiceStatus:
        url = f"{self._base_url}/health"
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                latency = (time.perf_counter() - start) * 1000
                body = json.loads(resp.read().decode())
                return ServiceStatus(
                    name=self.name,
                    status="healthy" if resp.status == 200 else "degraded",
                    latency_ms=round(latency, 2),
                    details=body,
                )
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            logger.warning("API health check failed: %s", exc)
            return ServiceStatus(
                name=self.name,
                status="down",
                latency_ms=round(latency, 2),
                details={"error": str(exc)},
            )


class RedisHealthChecker(BaseHealthChecker):
    """Checks Redis connectivity via a PING command."""
    name = "redis"

    def __init__(self):
        self._host = os.getenv("REDIS_HOST", "redis")
        self._port = int(os.getenv("REDIS_PORT", 6379))

    def check(self) -> ServiceStatus:
        start = time.perf_counter()
        try:
            import socket
            with socket.create_connection((self._host, self._port), timeout=5):
                latency = (time.perf_counter() - start) * 1000
                return ServiceStatus(
                    name=self.name,
                    status="healthy",
                    latency_ms=round(latency, 2),
                    details={"host": self._host, "port": self._port},
                )
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            logger.warning("Redis health check failed: %s", exc)
            return ServiceStatus(
                name=self.name,
                status="down",
                latency_ms=round(latency, 2),
                details={"error": str(exc)},
            )


class LLMHealthChecker(BaseHealthChecker):
    """Verifies LLM provider reachability (lightweight ping)."""
    name = "llm-provider"

    _PROVIDER_URLS: Dict[str, str] = {
        "openai": "https://api.openai.com",
        "anthropic": "https://api.anthropic.com",
        "azure_openai": "https://azure.microsoft.com",
    }

    def __init__(self):
        self._provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self._url = self._PROVIDER_URLS.get(self._provider, "https://api.openai.com")

    def check(self) -> ServiceStatus:
        start = time.perf_counter()
        try:
            req = urllib.request.Request(self._url, method="HEAD")
            with urllib.request.urlopen(req, timeout=8) as resp:
                latency = (time.perf_counter() - start) * 1000
                return ServiceStatus(
                    name=self.name,
                    status="healthy",
                    latency_ms=round(latency, 2),
                    details={"provider": self._provider, "url": self._url},
                )
        except urllib.error.HTTPError as exc:
            # HTTP errors (4xx/5xx) still mean the host is reachable
            latency = (time.perf_counter() - start) * 1000
            return ServiceStatus(
                name=self.name,
                status="healthy",
                latency_ms=round(latency, 2),
                details={"provider": self._provider, "http_code": exc.code},
            )
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            logger.warning("LLM health check failed: %s", exc)
            return ServiceStatus(
                name=self.name,
                status="degraded",
                latency_ms=round(latency, 2),
                details={"error": str(exc)},
            )


class EbayProviderChecker(BaseHealthChecker):
    """Lightweight check for eBay API availability."""
    name = "ebay-provider"

    def check(self) -> ServiceStatus:
        start = time.perf_counter()
        try:
            req = urllib.request.Request(
                "https://api.ebay.com", method="HEAD"
            )
            with urllib.request.urlopen(req, timeout=8):
                latency = (time.perf_counter() - start) * 1000
                return ServiceStatus(
                    name=self.name, status="healthy",
                    latency_ms=round(latency, 2),
                )
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            return ServiceStatus(
                name=self.name, status="degraded",
                latency_ms=round(latency, 2),
                details={"error": str(exc)},
            )


class AmazonProviderChecker(BaseHealthChecker):
    """Lightweight check for Amazon/RapidAPI availability."""
    name = "amazon-provider"

    def check(self) -> ServiceStatus:
        start = time.perf_counter()
        try:
            req = urllib.request.Request(
                "https://amazon.com", method="HEAD"
            )
            with urllib.request.urlopen(req, timeout=8):
                latency = (time.perf_counter() - start) * 1000
                return ServiceStatus(
                    name=self.name, status="healthy",
                    latency_ms=round(latency, 2),
                )
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            return ServiceStatus(
                name=self.name, status="degraded",
                latency_ms=round(latency, 2),
                details={"error": str(exc)},
            )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Collector
# ─────────────────────────────────────────────────────────────────────────────

class MetricsCollector:
    """
    Collects scraper performance metrics.
    In production, this reads from a shared Redis counter or
    the application's internal metrics registry.
    For now it exposes a stub that teams can hook into.
    """

    def collect(self) -> List[ScraperMetrics]:
        """
        Returns scraper metrics. Replace the stub values with
        real reads from Redis / Prometheus when available.
        """
        # ── Attempt to read from the API metrics endpoint ──
        try:
            url = f"http://localhost:{os.getenv('API_PORT', 8000)}/metrics"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                return [ScraperMetrics(**m) for m in data.get("scrapers", [])]
        except Exception:
            pass  # Fall back to stubs

        # ── Stub metrics (replaced by real data in production) ──
        return [
            ScraperMetrics(
                provider="ebay",
                total_requests=100, successful_requests=95, failed_requests=5,
                avg_latency_ms=320.5, listings_fetched=480,
            ),
            ScraperMetrics(
                provider="amazon",
                total_requests=80, successful_requests=76, failed_requests=4,
                avg_latency_ms=410.2, listings_fetched=380,
            ),
            ScraperMetrics(
                provider="walmart",
                total_requests=40, successful_requests=38, failed_requests=2,
                avg_latency_ms=280.0, listings_fetched=190,
            ),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Alert Engine
# ─────────────────────────────────────────────────────────────────────────────

class AlertEngine:
    """Evaluates the report and fires webhook alerts when thresholds are breached."""

    def __init__(self, webhook_url: Optional[str] = None):
        self._webhook_url = webhook_url or os.getenv("HEALTH_REPORT_WEBHOOK_URL", "")
        self._error_threshold = float(
            os.getenv("ALERT_THRESHOLD_ERROR_RATE", 0.10)
        )
        self._alert_enabled = os.getenv("ALERT_ON_ERROR", "true").lower() == "true"

    # ── Alert detection ───────────────────────────────────────────────────────

    def evaluate(self, report: HealthReport) -> List[str]:
        """Return a list of human-readable alert messages."""
        alerts: List[str] = []

        for svc in report.services:
            if svc.status == "down":
                alerts.append(f"🔴 SERVICE DOWN: {svc.name} is unreachable.")
            elif svc.status == "degraded":
                alerts.append(f"🟡 SERVICE DEGRADED: {svc.name} is experiencing issues.")
            if svc.latency_ms and svc.latency_ms > 2000:
                alerts.append(
                    f"⚠️  HIGH LATENCY: {svc.name} latency = {svc.latency_ms:.0f} ms."
                )

        for m in report.scraper_metrics:
            if m.error_rate > self._error_threshold:
                alerts.append(
                    f"🟡 HIGH ERROR RATE: {m.provider} scraper "
                    f"error rate = {m.error_rate:.1%} "
                    f"(threshold = {self._error_threshold:.1%})."
                )

        return alerts

    # ── Webhook delivery ──────────────────────────────────────────────────────

    def send_webhook(self, alerts: List[str], report: HealthReport) -> None:
        if not self._alert_enabled or not self._webhook_url or not alerts:
            return

        payload = {
            "text": f"*PricePulse Health Alert* — {report.generated_at}",
            "attachments": [
                {
                    "color": "danger" if report.overall_status == "critical" else "warning",
                    "text": "\n".join(alerts),
                    "footer": f"Environment: {report.environment} | "
                              f"Version: {report.app_version}",
                }
            ],
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            self._webhook_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status not in (200, 204):
                    raise WebhookDeliveryError(
                        f"Webhook returned HTTP {resp.status}"
                    )
            logger.info("Alert webhook delivered successfully.")
        except Exception as exc:
            raise WebhookDeliveryError(f"Failed to deliver webhook: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Report Renderer
# ─────────────────────────────────────────────────────────────────────────────

class ReportRenderer:
    """Writes HealthReport objects to disk as JSON and/or HTML."""

    STATUS_EMOJI = {
        "healthy": "✅",
        "degraded": "🟡",
        "down": "🔴",
        "critical": "🔴",
        "unknown": "❓",
    }

    def __init__(self, output_dir: str = "/app/reports"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON ──────────────────────────────────────────────────────────────────

    def save_json(self, report: HealthReport) -> Path:
        filename = self._output_dir / f"health_{report.report_id}.json"
        try:
            filename.write_text(report.to_json(), encoding="utf-8")
            logger.info("JSON report saved → %s", filename)
            return filename
        except OSError as exc:
            raise ReportGenerationError(f"Cannot write JSON report: {exc}") from exc

    # ── HTML ──────────────────────────────────────────────────────────────────

    def save_html(self, report: HealthReport) -> Path:
        filename = self._output_dir / f"health_{report.report_id}.html"
        html = self._build_html(report)
        try:
            filename.write_text(html, encoding="utf-8")
            logger.info("HTML report saved → %s", filename)
            return filename
        except OSError as exc:
            raise ReportGenerationError(f"Cannot write HTML report: {exc}") from exc

    def _build_html(self, report: HealthReport) -> str:
        status_color = {
            "healthy": "#2ecc71",
            "degraded": "#f39c12",
            "critical": "#e74c3c",
            "down": "#e74c3c",
        }.get(report.overall_status, "#95a5a6")

        STATUS_COLORS = {"healthy": "#2ecc71", "degraded": "#f39c12", "down": "#e74c3c"}
        services_rows = "".join(
            "<tr>"
            f"<td>{s.name}</td>"
            f'<td style="color:{STATUS_COLORS.get(s.status, "gray")}">'
            f"{self.STATUS_EMOJI.get(s.status, '')} {s.status.upper()}</td>"
            f"<td>{s.latency_ms:.1f} ms</td>"
            f"<td>{s.checked_at}</td>"
            "</tr>"
            for s in report.services
        )

        scraper_rows = "".join(
            f"""<tr>
                <td>{m.provider.upper()}</td>
                <td>{m.total_requests}</td>
                <td>{m.successful_requests}</td>
                <td>{m.failed_requests}</td>
                <td>{m.error_rate:.1%}</td>
                <td>{m.avg_latency_ms:.1f} ms</td>
                <td>{m.listings_fetched}</td>
            </tr>"""
            for m in report.scraper_metrics
        )

        alerts_html = (
            "<ul>" + "".join(f"<li>{a}</li>" for a in report.alerts) + "</ul>"
            if report.alerts
            else "<p style='color:#2ecc71'>No active alerts 🎉</p>"
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PricePulse Health Report — {report.report_id}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; padding: 24px; }}
    h1 {{ font-size: 1.8rem; margin-bottom: 4px; }}
    .subtitle {{ color: #888; font-size: .9rem; margin-bottom: 24px; }}
    .badge {{
      display: inline-block; padding: 6px 16px; border-radius: 20px;
      font-weight: bold; font-size: 1rem; color: #fff;
      background: {status_color}; margin-bottom: 24px;
    }}
    .card {{
      background: #1a1d27; border: 1px solid #2a2d3e;
      border-radius: 10px; padding: 20px; margin-bottom: 20px;
    }}
    .card h2 {{ font-size: 1.1rem; color: #a0a8c0; margin-bottom: 14px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
    th {{ text-align: left; padding: 8px 12px; background: #12141e; color: #7a80a0; border-bottom: 1px solid #2a2d3e; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #1e2130; }}
    tr:last-child td {{ border-bottom: none; }}
    .alerts {{ background: #1e1520; border-color: #4a2040; }}
    .alerts li {{ padding: 5px 0; }}
    .sysinfo {{ display: flex; flex-wrap: wrap; gap: 12px; }}
    .sysinfo-item {{ background: #12141e; border-radius: 6px; padding: 10px 16px; min-width: 160px; }}
    .sysinfo-item .label {{ font-size: .75rem; color: #7a80a0; margin-bottom: 4px; }}
    .sysinfo-item .value {{ font-size: .95rem; font-weight: 600; }}
    footer {{ margin-top: 30px; text-align: center; color: #444; font-size: .8rem; }}
  </style>
</head>
<body>
  <h1>🔍 PricePulse Health Report</h1>
  <div class="subtitle">
    Report ID: <code>{report.report_id}</code> |
    Generated: {report.generated_at} |
    Env: <strong>{report.environment}</strong> |
    Version: {report.app_version}
  </div>
  <div class="badge">{self.STATUS_EMOJI.get(report.overall_status,'')} {report.overall_status.upper()}</div>
  <p style="margin-bottom:24px;color:#aaa">{report.summary}</p>

  <!-- Services -->
  <div class="card">
    <h2>🛰 Service Status</h2>
    <table>
      <thead><tr><th>Service</th><th>Status</th><th>Latency</th><th>Checked At</th></tr></thead>
      <tbody>{services_rows}</tbody>
    </table>
  </div>

  <!-- Scraper Metrics -->
  <div class="card">
    <h2>📊 Scraper Metrics</h2>
    <table>
      <thead>
        <tr><th>Provider</th><th>Total Req.</th><th>Success</th><th>Failed</th>
            <th>Error Rate</th><th>Avg Latency</th><th>Listings</th></tr>
      </thead>
      <tbody>{scraper_rows}</tbody>
    </table>
  </div>

  <!-- Alerts -->
  <div class="card alerts">
    <h2>🚨 Alerts</h2>
    {alerts_html}
  </div>

  <!-- System Info -->
  <div class="card">
    <h2>🖥 System Info</h2>
    <div class="sysinfo">
      {"".join(
        f'<div class="sysinfo-item"><div class="label">{k.replace("_"," ").title()}</div><div class="value">{v}</div></div>'
        for k, v in report.system_info.items()
      )}
    </div>
  </div>

  <footer>PricePulse AI · SALIM Ayoub (DevOps/Reporting) · {report.generated_at}</footer>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator: HealthReportGenerator
# ─────────────────────────────────────────────────────────────────────────────

class HealthReportGenerator:
    """
    Orchestrates health checking, metric collection, alert evaluation,
    and report rendering for the PricePulse system.

    Composition:
        - List[BaseHealthChecker]  → service probes
        - MetricsCollector         → scraper KPIs
        - AlertEngine              → threshold evaluation + webhook
        - ReportRenderer           → JSON / HTML output
    """

    def __init__(
        self,
        output_dir: str | None = None,
        report_format: str | None = None,
        api_base_url: str = "http://localhost:8000",
    ):
        self._output_dir = output_dir or os.getenv("REPORT_OUTPUT_DIR", "/app/reports")
        self._report_format = (
            report_format or os.getenv("REPORT_FORMAT", "both")
        ).lower()

        # Compose dependencies
        self._checkers: List[BaseHealthChecker] = [
            APIHealthChecker(api_base_url),
            RedisHealthChecker(),
            LLMHealthChecker(),
            EbayProviderChecker(),
            AmazonProviderChecker(),
        ]
        self._metrics_collector = MetricsCollector()
        self._alert_engine = AlertEngine()
        self._renderer = ReportRenderer(self._output_dir)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self) -> HealthReport:
        """Run all checks and produce a complete HealthReport."""
        logger.info("Starting health report generation …")
        report_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        # 1. Run all service health checks
        services = [checker.check() for checker in self._checkers]

        # 2. Collect scraper metrics
        scraper_metrics = self._metrics_collector.collect()

        # 3. Determine overall status
        overall_status = self._compute_overall_status(services, scraper_metrics)

        # 4. System info
        system_info = self._collect_system_info()

        # 5. Build report object
        report = HealthReport(
            report_id=report_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            app_version=os.getenv("APP_VERSION", "1.0.0"),
            environment=os.getenv("APP_ENV", "development"),
            overall_status=overall_status,
            services=services,
            scraper_metrics=scraper_metrics,
            system_info=system_info,
        )

        # 6. Evaluate alerts
        report.alerts = self._alert_engine.evaluate(report)
        report.summary = self._build_summary(report)

        # 7. Persist report
        self._persist(report)

        # 8. Send alerts if any
        try:
            self._alert_engine.send_webhook(report.alerts, report)
        except WebhookDeliveryError as exc:
            logger.warning("Webhook delivery failed: %s", exc)

        logger.info(
            "Health report [%s] generated — overall: %s — alerts: %d",
            report_id, overall_status, len(report.alerts),
        )
        return report

    def run_scheduled(self) -> None:
        """Run generate() on a fixed interval (blocking loop)."""
        interval = int(os.getenv("REPORT_SCHEDULE_MINUTES", 60)) * 60
        logger.info("Health reporter scheduled every %d seconds.", interval)
        while True:
            try:
                self.generate()
            except Exception as exc:
                logger.error("Unexpected error during report generation: %s", exc)
            time.sleep(interval)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_overall_status(
        self,
        services: List[ServiceStatus],
        metrics: List[ScraperMetrics],
    ) -> str:
        down_count = sum(1 for s in services if s.status == "down")
        degraded_count = sum(1 for s in services if s.status == "degraded")
        error_threshold = float(os.getenv("ALERT_THRESHOLD_ERROR_RATE", 0.10))
        high_error = any(m.error_rate > error_threshold for m in metrics)

        if down_count >= 2 or (down_count == 1 and high_error):
            return "critical"
        if down_count == 1 or degraded_count >= 2 or high_error:
            return "degraded"
        return "healthy"

    def _collect_system_info(self) -> Dict[str, Any]:
        try:
            import psutil
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            cpu = psutil.cpu_percent(interval=1)
            return {
                "python_version": platform.python_version(),
                "os": f"{platform.system()} {platform.release()}",
                "cpu_usage_pct": f"{cpu}%",
                "memory_used_pct": f"{mem.percent}%",
                "disk_used_pct": f"{disk.percent}%",
                "hostname": platform.node(),
            }
        except ImportError:
            return {
                "python_version": platform.python_version(),
                "os": f"{platform.system()} {platform.release()}",
                "hostname": platform.node(),
                "note": "Install psutil for full system metrics",
            }

    def _build_summary(self, report: HealthReport) -> str:
        healthy = sum(1 for s in report.services if s.is_healthy)
        total = len(report.services)
        total_listings = sum(m.listings_fetched for m in report.scraper_metrics)
        return (
            f"{healthy}/{total} services healthy. "
            f"{len(report.alerts)} active alert(s). "
            f"{total_listings} listings fetched across all scrapers."
        )

    def _persist(self, report: HealthReport) -> None:
        if self._report_format in ("json", "both"):
            self._renderer.save_json(report)
        if self._report_format in ("html", "both"):
            self._renderer.save_html(report)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI endpoint helper (imported by app/main.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_health_status() -> Dict[str, Any]:
    """
    Lightweight health dict for the GET /health FastAPI endpoint.
    Does NOT run full checkers (too slow for a liveness probe).
    """
    return {
        "status": "healthy",
        "app": os.getenv("APP_NAME", "PricePulse-AI"),
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "environment": os.getenv("APP_ENV", "development"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PricePulse Health Report Generator"
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run as a daemon and generate reports on a fixed schedule.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the report output directory.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "html", "both"],
        default=None,
        help="Report format (default: from .env REPORT_FORMAT).",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the PricePulse API.",
    )
    args = parser.parse_args()

    generator = HealthReportGenerator(
        output_dir=args.output_dir,
        report_format=args.format,
        api_base_url=args.api_url,
    )

    if args.schedule:
        generator.run_scheduled()
    else:
        report = generator.generate()
        print(report.to_json())


if __name__ == "__main__":
    main()
