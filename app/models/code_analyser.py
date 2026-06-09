import requests
import logging
import time
from app.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeAnalyser:
    """
    Model 2: Analyses suspicious code snippets and scripts
    for malicious patterns, vulnerabilities and attack techniques.

    Uses Mistral running locally via Ollama — keeping sensitive
    code samples off external APIs, which is critical in a
    cybersecurity context.
    """

    def __init__(self):
        self.model_name = config.OLLAMA_CODE_MODEL
        self.base_url = config.OLLAMA_BASE_URL
        self.endpoint = f"{self.base_url}/api/generate"

    def _is_ollama_running(self) -> bool:
        """Check if Ollama server is accessible."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def _build_prompt(self, code: str) -> str:
        """
        Builds a structured prompt for security focused code analysis.
        Prompt engineering is documented here for the report.
        """
        return (
            "You are a cybersecurity expert analysing potentially malicious code.\n"
            "Analyse the following code snippet for security threats.\n\n"
            "Code to analyse:\n"
            + code +
            "\n\nProvide your analysis in exactly this format:\n"
            "THREAT_TYPE: [type of threat or NONE if benign]\n"
            "RISK_LEVEL: [CRITICAL/HIGH/MEDIUM/LOW/SAFE]\n"
            "CONFIDENCE: [HIGH/MEDIUM/LOW]\n"
            "TECHNIQUES: [comma separated list of techniques used]\n"
            "EXPLANATION: [2-3 sentences explaining what the code does]\n"
            "INDICATORS: [specific lines or patterns that are suspicious, or NONE]\n"
            "RECOMMENDATION: [what action should be taken]"
        )

    def analyse(self, code: str) -> dict:
        """
        Analyses a code snippet for malicious patterns.

        Args:
            code: Source code string to analyse

        Returns:
            dict containing threat assessment from Mistral
        """
        if not code or not code.strip():
            return {
                "error": "Empty code provided",
                "risk_level": "UNKNOWN",
                "model_used": self.model_name
            }

        if not self._is_ollama_running():
            return {
                "error": "Ollama server not running. Start with: ollama serve",
                "risk_level": "UNKNOWN",
                "model_used": self.model_name
            }

        try:
            logger.info(f"Analysing code snippet with {self.model_name}")
            time.sleep(2)

            prompt = self._build_prompt(code)

            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 500
                    }
                },
                timeout=120
            )

            if response.status_code != 200:
                return {
                    "error": f"Ollama returned status {response.status_code}",
                    "risk_level": "UNKNOWN",
                    "model_used": self.model_name
                }

            raw_response = response.json().get("response", "")
            parsed = self._parse_response(raw_response)
            parsed["model_used"] = self.model_name
            parsed["raw_response"] = raw_response
            parsed["code_preview"] = (
                code[:200] + "..." if len(code) > 200 else code
            )

            return parsed

        except requests.exceptions.Timeout:
            return {
                "error": "Request timed out — model may still be loading",
                "risk_level": "UNKNOWN",
                "model_used": self.model_name
            }
        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            return {
                "error": str(e),
                "risk_level": "UNKNOWN",
                "model_used": self.model_name
            }

    def _parse_response(self, response: str) -> dict:
        """
        Parses the structured response from Mistral
        into a clean dictionary. Strips any safety preamble
        before parsing structured fields.
        """
        # Find earliest field marker and strip everything before it
        field_markers = [
            "THREAT_TYPE:",
            "RISK_LEVEL:",
            "threat_type:",
            "risk_level:",
        ]

        earliest_pos = len(response)
        for marker in field_markers:
            pos = response.find(marker)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos

        if earliest_pos < len(response):
            response = response[earliest_pos:]

        result = {
            "threat_type": "UNKNOWN",
            "risk_level": "UNKNOWN",
            "confidence": "LOW",
            "techniques": [],
            "explanation": "",
            "indicators": "",
            "recommendation": ""
        }

        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith("THREAT_TYPE:"):
                result["threat_type"] = line.replace(
                    "THREAT_TYPE:", ""
                ).strip()
            elif line.startswith("RISK_LEVEL:"):
                result["risk_level"] = line.replace(
                    "RISK_LEVEL:", ""
                ).strip()
            elif line.startswith("CONFIDENCE:"):
                result["confidence"] = line.replace(
                    "CONFIDENCE:", ""
                ).strip()
            elif line.startswith("TECHNIQUES:"):
                techniques_str = line.replace("TECHNIQUES:", "").strip()
                result["techniques"] = [
                    t.strip() for t in techniques_str.split(",")
                ]
            elif line.startswith("EXPLANATION:"):
                result["explanation"] = line.replace(
                    "EXPLANATION:", ""
                ).strip()
            elif line.startswith("INDICATORS:"):
                result["indicators"] = line.replace(
                    "INDICATORS:", ""
                ).strip()
            elif line.startswith("RECOMMENDATION:"):
                result["recommendation"] = line.replace(
                    "RECOMMENDATION:", ""
                ).strip()

        return result

    def evaluate_model(self, test_cases: list) -> dict:
        """
        Evaluates model performance on test cases.
        Used for model selection documentation in the report.
        """
        results = []
        correct = 0

        for case in test_cases:
            result = self.analyse(case["code"])
            expected = case["expected_risk"]
            actual = result.get("risk_level", "UNKNOWN")

            is_correct = actual.upper() == expected.upper()
            if is_correct:
                correct += 1

            results.append({
                "category": case.get("category", "Unknown"),
                "expected": expected,
                "actual": actual,
                "correct": is_correct,
                "threat_type": result.get("threat_type", ""),
                "explanation": result.get("explanation", "")[:100]
            })

        accuracy = correct / len(test_cases) if test_cases else 0

        return {
            "model": self.model_name,
            "total_cases": len(test_cases),
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "results": results
        }


# Single instance used across the application
code_analyser = CodeAnalyser()