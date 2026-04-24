import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.pipeline import CyberSentinelPipeline


class TestCyberSentinelPipeline:
    """
    Integration tests for the CyberSentinel Pipeline.
    Tests the orchestration of all three models working together.
    """

    @pytest.fixture
    def pipeline(self):
        return CyberSentinelPipeline()

    def test_pipeline_initialises(self, pipeline):
        """Test that pipeline initialises with all three models."""
        assert pipeline.threat_classifier is not None
        assert pipeline.code_analyser is not None
        assert pipeline.urgency_classifier is not None

    def test_empty_input_returns_error(self, pipeline):
        """Test that no input returns an error."""
        result = pipeline.analyse()
        assert "error" in result

    def test_log_only_analysis(self, pipeline):
        """Test pipeline with only log text input."""
        result = pipeline.analyse(
            log_text="Failed login attempt from suspicious IP"
        )
        assert "overall_risk" in result
        assert "summary" in result
        assert result["threat_analysis"] is not None
        assert result["code_analysis"] is None
        assert result["urgency_analysis"] is None

    def test_bulletin_only_analysis(self, pipeline):
        """Test pipeline with only bulletin text input."""
        result = pipeline.analyse(
            bulletin_text="Critical zero-day vulnerability actively exploited"
        )
        assert "overall_urgency" in result
        assert result["urgency_analysis"] is not None
        assert result["threat_analysis"] is None

    def test_full_pipeline_all_inputs(self, pipeline):
        """Test pipeline with all three inputs simultaneously."""
        result = pipeline.analyse(
            log_text="Unauthorized access detected from IP 45.33.32.156",
            code_snippet="import os\nos.system('rm -rf /')",
            bulletin_text="Critical vulnerability actively exploited in wild"
        )
        assert result["threat_analysis"] is not None
        assert result["code_analysis"] is not None
        assert result["urgency_analysis"] is not None
        assert len(result["models_used"]) == 3

    def test_overall_risk_is_valid(self, pipeline):
        """Test that overall risk is a valid value."""
        valid_risks = {
            "CRITICAL", "HIGH", "MEDIUM", "LOW",
            "SAFE", "INFORMATIONAL", "UNKNOWN"
        }
        result = pipeline.analyse(
            log_text="Malware detected on endpoint"
        )
        assert result["overall_risk"] in valid_risks

    def test_timestamp_present(self, pipeline):
        """Test that timestamp is included in results."""
        result = pipeline.analyse(
            log_text="Security event detected"
        )
        assert "timestamp" in result
        assert result["timestamp"] is not None

    def test_models_used_tracked(self, pipeline):
        """Test that models used are tracked in results."""
        result = pipeline.analyse(
            log_text="Suspicious activity detected",
            bulletin_text="Security advisory issued"
        )
        assert isinstance(result["models_used"], list)
        assert len(result["models_used"]) == 2

    def test_recommendations_generated(self, pipeline):
        """Test that recommendations are generated."""
        result = pipeline.analyse(
            log_text="Malware signature detected in uploaded file",
            bulletin_text="Ransomware actively targeting organisations"
        )
        assert isinstance(result["recommendations"], list)

    def test_summary_is_string(self, pipeline):
        """Test that summary is a non-empty string."""
        result = pipeline.analyse(
            log_text="Failed authentication attempt detected"
        )
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_highest_risk_selected(self, pipeline):
        """Test that overall risk reflects highest severity found."""
        result = pipeline.analyse(
            log_text="Critical malware detected attempting data exfiltration",
            bulletin_text="Zero-day exploit actively used by threat actors"
        )
        # With critical inputs, overall risk should be high severity
        assert result["overall_risk"] in {"CRITICAL", "HIGH", "MEDIUM"}