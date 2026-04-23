import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.threat_classifier import ThreatClassifier

class TestThreatClassifier:
    """
    Unit tests for the ThreatClassifier model.
    Tests cover model loading, analysis output structure,
    risk level mapping, and edge cases.
    """
    
    @pytest.fixture
    def classifier(self):
        """Create a fresh classifier instance for each test."""
        clf = ThreatClassifier()
        clf.load()
        return clf
    
    def test_model_loads_successfully(self, classifier):
        """Test that model loads without errors."""
        assert classifier.is_loaded == True
        assert classifier.classifier is not None
    
    def test_analyse_returns_required_fields(self, classifier):
        """Test that analysis output contains all required fields."""
        result = classifier.analyse("Failed login attempt from unknown IP")
        
        assert "label" in result
        assert "confidence" in result
        assert "risk_level" in result
        assert "model_used" in result
        assert "flagged" in result
    
    def test_confidence_is_between_0_and_1(self, classifier):
        """Test that confidence score is a valid probability."""
        result = classifier.analyse("Malware detected in system files")
        assert 0 <= result["confidence"] <= 1
    
    def test_risk_level_is_valid_value(self, classifier):
        """Test that risk level is one of the expected values."""
        valid_levels = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "SAFE", "UNKNOWN"}
        result = classifier.analyse("Unauthorized access attempt detected")
        assert result["risk_level"] in valid_levels
    
    def test_empty_input_handled_gracefully(self, classifier):
        """Test that empty input doesn't crash the system."""
        result = classifier.analyse("")
        assert "error" in result
        assert result["risk_level"] == "UNKNOWN"
    
    def test_suspicious_text_flagged(self, classifier):
        """Test that clearly suspicious text gets flagged."""
        result = classifier.analyse(
            "SQL injection attack detected: malicious payload in request"
        )
        assert result["risk_level"] in {"CRITICAL", "HIGH", "MEDIUM"}
    
    def test_benign_text_lower_risk(self, classifier):
        """Test that normal system logs receive lower risk ratings."""
        result = classifier.analyse(
            "System backup completed successfully"
        )
        assert result["risk_level"] in {"SAFE", "LOW", "MEDIUM"}
    
    def test_evaluate_model_returns_metrics(self, classifier):
        """Test that model evaluation produces accuracy metrics."""
        test_cases = [
            {
                "text": "Malware detected attempting to exfiltrate data",
                "expected_risk": "CRITICAL"
            },
            {
                "text": "User login successful from trusted network",
                "expected_risk": "SAFE"
            }
        ]
        
        evaluation = classifier.evaluate_model(test_cases)
        
        assert "accuracy" in evaluation
        assert "total_cases" in evaluation
        assert "model" in evaluation
        assert evaluation["total_cases"] == 2
    
    def test_long_input_handled(self, classifier):
        """Test that very long inputs are handled without errors."""
        long_text = "Failed login attempt. " * 100
        result = classifier.analyse(long_text)
        assert "risk_level" in result
        assert "error" not in result


class TestThreatClassifierEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    @pytest.fixture
    def classifier(self):
        clf = ThreatClassifier()
        clf.load()
        return clf
    
    def test_whitespace_only_input(self, classifier):
        """Test that whitespace-only input is handled."""
        result = classifier.analyse("   ")
        assert result["risk_level"] == "UNKNOWN"
    
    def test_special_characters_handled(self, classifier):
        """Test that special characters don't break the model."""
        result = classifier.analyse("SELECT * FROM users; DROP TABLE users;--")
        assert "risk_level" in result
    
    def test_model_name_recorded(self, classifier):
        """Test that model name is recorded for report documentation."""
        result = classifier.analyse("Test input")
        assert result["model_used"] == classifier.model_name