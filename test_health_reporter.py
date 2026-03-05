

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Set dummy env vars before importing the module
os.environ.setdefault("APP_VERSION", "1.0.0")
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("LLM_PROVIDER", "openai")

from app.reporting.health_reporter import (
    AlertEngine,
    HealthReport,
    HealthReportGenerator,
    MetricsCollector,
    ReportGenerationError,
    ReportRenderer,
    ScraperMetrics,
    ServiceStatus,
    WebhookDeliveryError,
    get_health_status,
)




@pytest.fixture
def healthy_service():
    return ServiceStatus(
        name="pricepulse-api", status="healthy", latency_ms=42.0
    )


@pytest.fixture
def down_service():
    return ServiceStatus(
        name="redis", status="down", latency_ms=5000.0,
        details={"error": "Connection refused"}
    )


@pytest.fixture
def scraper_metrics_ok():
    return ScraperMetrics(
        provider="ebay",
        total_requests=100, successful_requests=98, failed_requests=2,
        avg_latency_ms=310.0, listings_fetched=490,
    )


@pytest.fixture
def scraper_metrics_high_error():
    return ScraperMetrics(
        provider="amazon",
        total_requests=100, successful_requests=70, failed_requests=30,
        avg_latency_ms=800.0, listings_fetched=300,
    )


@pytest.fixture
def sample_report(healthy_service, scraper_metrics_ok):
    return HealthReport(
        report_id="20240101T120000Z",
        generated_at="2024-01-01T12:00:00+00:00",
        app_version="1.0.0",
        environment="test",
        overall_status="healthy",
        services=[healthy_service],
        scraper_metrics=[scraper_metrics_ok],
        system_info={"python_version": "3.11.0"},
        alerts=[],
        summary="All good.",
    )


@pytest.fixture
def tmp_output_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d




class TestServiceStatus:
    def test_is_healthy_true(self, healthy_service):
        assert healthy_service.is_healthy is True

    def test_is_healthy_false_when_down(self, down_service):
        assert down_service.is_healthy is False

    def test_checked_at_is_set(self, healthy_service):
        assert healthy_service.checked_at is not None



class TestScraperMetrics:
    def test_error_rate_ok(self, scraper_metrics_ok):
        assert scraper_metrics_ok.error_rate == pytest.approx(0.02)

    def test_success_rate_ok(self, scraper_metrics_ok):
        assert scraper_metrics_ok.success_rate == pytest.approx(0.98)

    def test_error_rate_high(self, scraper_metrics_high_error):
        assert scraper_metrics_high_error.error_rate == pytest.approx(0.30)

    def test_error_rate_zero_requests(self):
        m = ScraperMetrics(provider="walmart", total_requests=0)
        assert m.error_rate == 0.0




class TestHealthReport:
    def test_to_dict_has_required_keys(self, sample_report):
        d = sample_report.to_dict()
        for key in ("report_id", "generated_at", "overall_status", "services",
                    "scraper_metrics", "alerts"):
            assert key in d

    def test_to_json_is_valid(self, sample_report):
        raw = sample_report.to_json()
        parsed = json.loads(raw)
        assert parsed["report_id"] == "20240101T120000Z"



class TestAlertEngine:
    def test_no_alerts_when_healthy(self, sample_report, scraper_metrics_ok):
        engine = AlertEngine()
        alerts = engine.evaluate(sample_report)
        assert alerts == []

    def test_alert_when_service_down(self, sample_report, down_service):
        sample_report.services.append(down_service)
        engine = AlertEngine()
        alerts = engine.evaluate(sample_report)
        assert any("DOWN" in a for a in alerts)

    def test_alert_when_high_error_rate(self, sample_report, scraper_metrics_high_error):
        sample_report.scraper_metrics.append(scraper_metrics_high_error)
        engine = AlertEngine()
        alerts = engine.evaluate(sample_report)
        assert any("amazon" in a.lower() for a in alerts)

    def test_alert_high_latency(self, sample_report):
        slow = ServiceStatus(name="slow-svc", status="healthy", latency_ms=3000.0)
        sample_report.services.append(slow)
        engine = AlertEngine()
        alerts = engine.evaluate(sample_report)
        assert any("LATENCY" in a for a in alerts)

    def test_webhook_skipped_when_no_url(self, sample_report):
        engine = AlertEngine(webhook_url="")
        # Should not raise
        engine.send_webhook(["test alert"], sample_report)

    def test_webhook_raises_on_failure(self, sample_report):
        engine = AlertEngine(webhook_url="http://invalid.test/webhook")
        with pytest.raises(WebhookDeliveryError):
            engine.send_webhook(["alert"], sample_report)




class TestReportRenderer:
    def test_save_json_creates_file(self, sample_report, tmp_output_dir):
        renderer = ReportRenderer(output_dir=tmp_output_dir)
        path = renderer.save_json(sample_report)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["report_id"] == sample_report.report_id

    def test_save_html_creates_file(self, sample_report, tmp_output_dir):
        renderer = ReportRenderer(output_dir=tmp_output_dir)
        path = renderer.save_html(sample_report)
        assert path.exists()
        content = path.read_text()
        assert "PricePulse" in content
        assert sample_report.report_id in content

    def test_save_json_raises_on_bad_dir(self, sample_report):
        renderer = ReportRenderer.__new__(ReportRenderer)
        renderer._output_dir = Path("/nonexistent_dir_xyz/deep/path")
        with pytest.raises(ReportGenerationError):
            renderer.save_json(sample_report)



class TestMetricsCollector:
    def test_returns_list_of_scraper_metrics(self):
        collector = MetricsCollector()
        metrics = collector.collect()
        assert isinstance(metrics, list)
        assert all(isinstance(m, ScraperMetrics) for m in metrics)

    def test_at_least_two_providers(self):
        collector = MetricsCollector()
        metrics = collector.collect()
        providers = {m.provider for m in metrics}
        assert len(providers) >= 2




class TestHealthReportGenerator:
    def test_generate_returns_health_report(self, tmp_output_dir):
        gen = HealthReportGenerator(
            output_dir=tmp_output_dir,
            report_format="json",
            api_base_url="http://localhost:9999",  # unreachable → "down"
        )
        report = gen.generate()
        assert isinstance(report, HealthReport)
        assert report.overall_status in ("healthy", "degraded", "critical")

    def test_reports_are_persisted(self, tmp_output_dir):
        gen = HealthReportGenerator(
            output_dir=tmp_output_dir,
            report_format="both",
            api_base_url="http://localhost:9999",
        )
        report = gen.generate()
        files = list(Path(tmp_output_dir).iterdir())
        assert any(f.suffix == ".json" for f in files)
        assert any(f.suffix == ".html" for f in files)

    def test_compute_overall_status_healthy(self):
        gen = HealthReportGenerator.__new__(HealthReportGenerator)
        services = [ServiceStatus("svc", "healthy", 50.0)]
        metrics = [ScraperMetrics("ebay", 100, 99, 1, 200.0, 400)]
        assert gen._compute_overall_status(services, metrics) == "healthy"

    def test_compute_overall_status_critical(self):
        gen = HealthReportGenerator.__new__(HealthReportGenerator)
        services = [
            ServiceStatus("svc1", "down", 0.0),
            ServiceStatus("svc2", "down", 0.0),
        ]
        metrics = []
        assert gen._compute_overall_status(services, metrics) == "critical"




class TestGetHealthStatus:
    def test_returns_dict_with_status(self):
        result = get_health_status()
        assert result["status"] == "healthy"
        assert "version" in result
        assert "timestamp" in result
