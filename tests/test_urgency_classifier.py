import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.urgency_classifier import UrgencyClassifier


class TestUrgencyClassifier:
    """
    Unit tests for the UrgencyClassifier model.
    Tests cover model loading, urgency mapping,
    batch analysis and edge cases.
    """

    @pytest.fixture
    def classifier(self):
        clf = UrgencyClassifier()
        clf.load()
        return clf

    def test_model_loads_successfully(self, classifier):
        """Test that model loads without errors."""
        assert classifier.is_loaded == True
        assert classifier.classifier is not None

    def test_analyse_returns_required_fields(self, classifier):
        """Test that analysis output contains all required fields."""
        result = classifier.analyse(
            "Critical vulnerability detected in production system"
        )
        assert "urgency_level" in result
        assert "confidence" in result
        assert "priority" in result
        assert "response_time" in result
        assert "recommended_action" in result
        assert "model_used" in result

    def test_empty_input_handled(self, classifier):
        """Test that empty input is handled gracefully."""
        result = classifier.analyse("")
        assert "error" in result
        assert result["urgency_level"] == "UNKNOWN"

    def test_whitespace_input_handled(self, classifier):
        """Test that whitespace only input is handled gracefully."""
        result = classifier.analyse("   ")
        assert "error" in result
        assert result["urgency_level"] == "UNKNOWN"

    def test_critical_threat_detected(self, classifier):
        """Test that zero-day exploit is classified as critical."""
        result = classifier.analyse(
            "Zero-day vulnerability actively exploited in the wild. "
            "Remote code execution possible without authentication."
        )
        assert result["urgency_level"] in {"CRITICAL", "HIGH"}

    def test_ransomware_high_urgency(self, classifier):
        """Test that ransomware alert receives high urgency."""
        result = classifier.analyse(
            "Ransomware campaign detected targeting enterprise networks. "
            "Multiple systems already encrypted."
        )
        assert result["urgency_level"] in {"CRITICAL", "HIGH"}

    def test_informational_low_urgency(self, classifier):
        """Test that routine notices receive low urgency."""
        result = classifier.analyse(
            "Monthly security newsletter. General best practices reminder. "
            "No immediate action required."
        )
        assert result["urgency_level"] in {"LOW", "INFORMATIONAL"}

    def test_urgency_level_valid_value(self, classifier):
        """Test that urgency level is one of expected values."""
        valid_levels = {
            "CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL", "UNKNOWN"
        }
        result = classifier.analyse("Security advisory notice")
        assert result["urgency_level"] in valid_levels

    def test_priority_is_integer(self, classifier):
        """Test that priority is returned as an integer."""
        result = classifier.analyse(
            "Vulnerability detected in system"
        )
        if "error" not in result:
            assert isinstance(result["priority"], int)

    def test_confidence_between_0_and_1(self, classifier):
        """Test that confidence score is a valid probability."""
        result = classifier.analyse(
            "Security patch available for download"
        )
        if "error" not in result:
            assert 0 <= result["confidence"] <= 1

    def test_batch_analyse_returns_list(self, classifier):
        """Test that batch analysis returns a sorted list."""
        texts = [
            "Routine system maintenance scheduled",
            "Zero-day exploit actively being used by attackers",
            "Medium severity vulnerability patched"
        ]
        results = classifier.batch_analyse(texts)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_batch_sorted_by_priority(self, classifier):
        """Test that batch results are sorted most urgent first."""
        texts = [
            "Routine system maintenance scheduled",
            "Zero-day exploit actively being used by attackers",
        ]
        results = classifier.batch_analyse(texts)
        if len(results) >= 2:
            assert results[0]["priority"] <= results[1]["priority"]

    def test_model_name_recorded(self, classifier):
        """Test that model name is recorded for documentation."""
        result = classifier.analyse("Security alert detected")
        assert result["model_used"] == classifier.model_name

    def test_evaluate_model_returns_metrics(self, classifier):
        """Test that model evaluation produces accuracy metrics."""
        test_cases = [
            {
                "text": "Zero-day vulnerability actively exploited",
                "expected_urgency": "CRITICAL"
            },
            {
                "text": "Routine maintenance notification",
                "expected_urgency": "INFORMATIONAL"
            }
        ]
        evaluation = classifier.evaluate_model(test_cases)
        assert "accuracy" in evaluation
        assert "total_cases" in evaluation
        assert evaluation["total_cases"] == 2