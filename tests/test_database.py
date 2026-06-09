import pytest
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use a test database
os.environ["DB_PATH"] = "test_cybersentinel.db"

from app.database import (
    initialise_database,
    save_analysis,
    get_all_analyses,
    get_analysis_by_id,
    get_risk_statistics,
    delete_analysis
)

class TestDatabase:
    """Unit tests for the SQLite database layer."""

    def setup_method(self):
        """Initialise fresh database before each test."""
        initialise_database()

    def teardown_method(self):
        """Remove test database after each test."""
        if os.path.exists("test_cybersentinel.db"):
            os.remove("test_cybersentinel.db")

    def _sample_result(self, risk="CRITICAL"):
        return {
            "timestamp": "2026-06-09T12:00:00",
            "overall_risk": risk,
            "overall_urgency": "HIGH",
            "models_used": ["distilbert", "mistral"],
            "threat_analysis": {"risk_level": risk, "confidence": 0.99},
            "code_analysis": {"risk_level": risk, "threat_type": "Malware"},
            "urgency_analysis": {"urgency_level": "HIGH"},
            "recommendations": ["Investigate immediately"],
            "summary": "Test summary"
        }

    def _sample_inputs(self):
        return {
            "log_text": "Failed login attempt",
            "code_snippet": "import os",
            "bulletin_text": "Critical vulnerability"
        }

    def test_database_initialises(self):
        """Test database and tables are created."""
        assert os.path.exists("test_cybersentinel.db")

    def test_save_analysis_returns_id(self):
        """Test that saving returns a valid integer ID."""
        record_id = save_analysis(
            self._sample_result(), self._sample_inputs()
        )
        assert isinstance(record_id, int)
        assert record_id > 0

    def test_get_all_analyses_returns_list(self):
        """Test that history retrieval returns a list."""
        save_analysis(self._sample_result(), self._sample_inputs())
        analyses = get_all_analyses()
        assert isinstance(analyses, list)
        assert len(analyses) >= 1

    def test_saved_analysis_retrievable(self):
        """Test that a saved analysis can be retrieved by ID."""
        record_id = save_analysis(
            self._sample_result(), self._sample_inputs()
        )
        analysis = get_analysis_by_id(record_id)
        assert analysis is not None
        assert analysis["overall_risk"] == "CRITICAL"

    def test_get_analysis_nonexistent_returns_none(self):
        """Test that requesting a nonexistent ID returns None."""
        result = get_analysis_by_id(99999)
        assert result is None

    def test_risk_statistics_returns_counts(self):
        """Test that statistics are calculated correctly."""
        save_analysis(self._sample_result("CRITICAL"), self._sample_inputs())
        save_analysis(self._sample_result("HIGH"), self._sample_inputs())
        save_analysis(self._sample_result("CRITICAL"), self._sample_inputs())

        stats = get_risk_statistics()
        assert stats["total_analyses"] >= 3
        assert "by_risk_level" in stats
        assert stats["by_risk_level"].get("CRITICAL", 0) >= 2

    def test_delete_analysis(self):
        """Test that an analysis can be deleted."""
        record_id = save_analysis(
            self._sample_result(), self._sample_inputs()
        )
        deleted = delete_analysis(record_id)
        assert deleted == True
        assert get_analysis_by_id(record_id) is None

    def test_delete_nonexistent_returns_false(self):
        """Test that deleting nonexistent ID returns False."""
        result = delete_analysis(99999)
        assert result == False

    def test_multiple_analyses_saved(self):
        """Test that multiple analyses accumulate correctly."""
        for i in range(5):
            save_analysis(self._sample_result(), self._sample_inputs())

        analyses = get_all_analyses()
        assert len(analyses) >= 5

    def test_limit_parameter_respected(self):
        """Test that the limit parameter works correctly."""
        for i in range(10):
            save_analysis(self._sample_result(), self._sample_inputs())

        analyses = get_all_analyses(limit=3)
        assert len(analyses) <= 3