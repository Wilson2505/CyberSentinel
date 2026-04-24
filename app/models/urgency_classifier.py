from transformers import pipeline
from app.config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UrgencyClassifier:
    """
    Model 3: Analyses threat intelligence bulletins, CVE descriptions
    and security advisories to classify urgency and priority level.

    Uses a HuggingFace sentiment model to assess the urgency of
    threat intelligence text — helping analysts prioritise which
    threats require immediate response vs monitoring.
    """

    URGENCY_LEVELS = {
        "CRITICAL": {
            "priority": 1,
            "response_time": "Immediate — within 1 hour",
            "action": "Escalate to senior analyst and incident response team"
        },
        "HIGH": {
            "priority": 2,
            "response_time": "Urgent — within 4 hours",
            "action": "Assign to analyst for immediate investigation"
        },
        "MEDIUM": {
            "priority": 3,
            "response_time": "Standard — within 24 hours",
            "action": "Add to investigation queue"
        },
        "LOW": {
            "priority": 4,
            "response_time": "Monitoring — within 72 hours",
            "action": "Log and monitor for escalation"
        },
        "INFORMATIONAL": {
            "priority": 5,
            "response_time": "No immediate action required",
            "action": "File for reference"
        }
    }

    def __init__(self):
        self.model_name = config.HF_URGENCY_MODEL
        self.classifier = None
        self.is_loaded = False

    def load(self):
        """Load the urgency classification model."""
        try:
            logger.info(f"Loading urgency classifier: {self.model_name}")
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                truncation=True,
                max_length=512
            )
            self.is_loaded = True
            logger.info("Urgency classifier loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load urgency classifier: {e}")
            raise

    def analyse(self, text: str) -> dict:
        """
        Analyses threat intelligence text for urgency level.

        Args:
            text: CVE description, security bulletin, or threat advisory

        Returns:
            dict containing urgency assessment and recommended response
        """
        if not self.is_loaded:
            self.load()

        if not text or not text.strip():
            return {
                "error": "Empty input provided",
                "urgency_level": "UNKNOWN",
                "model_used": self.model_name
            }

        try:
            display_text = text[:200] + "..." if len(text) > 200 else text

            result = self.classifier(text)[0]

            label = result["label"]
            confidence = result["score"]

            urgency_level = self._map_to_urgency(label, confidence, text)
            urgency_details = self.URGENCY_LEVELS.get(urgency_level, {})

            return {
                "label": label,
                "confidence": round(confidence, 4),
                "urgency_level": urgency_level,
                "priority": urgency_details.get("priority", 99),
                "response_time": urgency_details.get(
                    "response_time", "Unknown"
                ),
                "recommended_action": urgency_details.get(
                    "action", "Review manually"
                ),
                "raw_text": display_text,
                "model_used": self.model_name
            }

        except Exception as e:
            logger.error(f"Urgency analysis failed: {e}")
            return {
                "error": str(e),
                "urgency_level": "UNKNOWN",
                "model_used": self.model_name
            }

    def _map_to_urgency(
        self, label: str, confidence: float, text: str
    ) -> str:
        """
        Maps model output to urgency levels using label,
        confidence score and keyword analysis.
        """
        # Check for critical keywords in the text
        critical_keywords = [
            "zero-day", "0day", "actively exploited", "ransomware",
            "critical vulnerability", "remote code execution", "rce",
            "data breach", "exfiltration", "backdoor"
        ]
        high_keywords = [
            "unauthorized access", "privilege escalation", "malware",
            "phishing", "sql injection", "denial of service", "dos",
            "vulnerability", "exploit", "attack detected"
        ]

        text_lower = text.lower()

        has_critical = any(kw in text_lower for kw in critical_keywords)
        has_high = any(kw in text_lower for kw in high_keywords)

        if label == "NEGATIVE":
            if has_critical or confidence >= 0.95:
                return "CRITICAL"
            elif has_high or confidence >= 0.80:
                return "HIGH"
            elif confidence >= 0.65:
                return "MEDIUM"
            else:
                return "LOW"
        else:
            # POSITIVE label
            if has_critical:
                return "HIGH"
            elif has_high:
                return "MEDIUM"
            else:
                return "INFORMATIONAL"

    def batch_analyse(self, texts: list) -> list:
        """
        Analyses multiple threat bulletins and returns
        them sorted by urgency priority.

        Args:
            texts: list of threat intelligence strings

        Returns:
            list of results sorted by priority (most urgent first)
        """
        if not self.is_loaded:
            self.load()

        results = []
        for text in texts:
            result = self.analyse(text)
            result["original_text"] = text[:100]
            results.append(result)

        # Sort by priority — lower number = more urgent
        results.sort(key=lambda x: x.get("priority", 99))

        return results

    def evaluate_model(self, test_cases: list) -> dict:
        """
        Evaluates model performance on test cases.
        Used for model selection documentation in the report.
        """
        if not self.is_loaded:
            self.load()

        results = []
        correct = 0

        for case in test_cases:
            result = self.analyse(case["text"])
            expected = case["expected_urgency"]
            actual = result.get("urgency_level", "UNKNOWN")
            is_correct = actual == expected

            if is_correct:
                correct += 1

            results.append({
                "text": case["text"][:80],
                "expected": expected,
                "actual": actual,
                "correct": is_correct,
                "confidence": result.get("confidence", 0)
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
urgency_classifier = UrgencyClassifier()