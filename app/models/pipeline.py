import logging
from datetime import datetime
from app.models.threat_classifier import threat_classifier
from app.models.code_analyser import code_analyser
from app.models.urgency_classifier import urgency_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CyberSentinelPipeline:
    """
    Core Pipeline: Orchestrates all three AI models to produce
    a unified threat assessment report.

    Data flow:
        Input (logs + code + bulletins)
            ↓
        Model 1: ThreatClassifier  → threat type + risk level
        Model 2: CodeAnalyser      → code vulnerability assessment
        Model 3: UrgencyClassifier → priority + response time
            ↓
        Synthesis: Combined threat report
            ↓
        Output: Structured assessment with recommendations
    """

    RISK_PRIORITY = {
        "CRITICAL": 1,
        "HIGH": 2,
        "MEDIUM": 3,
        "LOW": 4,
        "SAFE": 5,
        "INFORMATIONAL": 6,
        "UNKNOWN": 7
    }

    def __init__(self):
        self.threat_classifier = threat_classifier
        self.code_analyser = code_analyser
        self.urgency_classifier = urgency_classifier
        logger.info("CyberSentinel Pipeline initialised")

    def analyse(
        self,
        log_text: str = None,
        code_snippet: str = None,
        bulletin_text: str = None
    ) -> dict:
        """
        Runs all available inputs through their respective models
        and synthesises a unified threat assessment.

        Args:
            log_text:      Raw log entries or security report text
            code_snippet:  Suspicious code to analyse
            bulletin_text: CVE description or threat advisory

        Returns:
            dict containing unified threat assessment report
        """
        if not any([log_text, code_snippet, bulletin_text]):
            return {
                "error": "At least one input required",
                "timestamp": datetime.now().isoformat()
            }

        results = {
            "timestamp": datetime.now().isoformat(),
            "models_used": [],
            "threat_analysis": None,
            "code_analysis": None,
            "urgency_analysis": None,
            "overall_risk": "UNKNOWN",
            "overall_urgency": "UNKNOWN",
            "summary": "",
            "recommendations": []
        }

        # Run Model 1 — Threat Classifier
        if log_text and log_text.strip():
            logger.info("Running Model 1: Threat Classifier")
            try:
                threat_result = self.threat_classifier.analyse(log_text)
                results["threat_analysis"] = threat_result
                results["models_used"].append("distilbert-threat-classifier")
                logger.info(
                    f"Model 1 complete — Risk: {threat_result.get('risk_level')}"
                )
            except Exception as e:
                logger.error(f"Model 1 failed: {e}")
                results["threat_analysis"] = {"error": str(e)}

        # Run Model 2 — Code Analyser
        if code_snippet and code_snippet.strip():
            logger.info("Running Model 2: Code Analyser")
            try:
                code_result = self.code_analyser.analyse(code_snippet)
                results["code_analysis"] = code_result
                results["models_used"].append("codellama-code-analyser")
                logger.info(
                    f"Model 2 complete — Risk: {code_result.get('risk_level')}"
                )
            except Exception as e:
                logger.error(f"Model 2 failed: {e}")
                results["code_analysis"] = {"error": str(e)}

        # Run Model 3 — Urgency Classifier
        if bulletin_text and bulletin_text.strip():
            logger.info("Running Model 3: Urgency Classifier")
            try:
                urgency_result = self.urgency_classifier.analyse(bulletin_text)
                results["urgency_analysis"] = urgency_result
                results["models_used"].append(
                    "multilingual-sentiment-urgency-classifier"
                )
                logger.info(
                    f"Model 3 complete — Urgency: {urgency_result.get('urgency_level')}"
                )
            except Exception as e:
                logger.error(f"Model 3 failed: {e}")
                results["urgency_analysis"] = {"error": str(e)}

        # Synthesise results
        results = self._synthesise(results)

        return results

    def _synthesise(self, results: dict) -> dict:
        """
        Combines outputs from all three models into a unified
        risk assessment and recommendation set.
        """
        risk_levels = []
        recommendations = []

        # Collect risk levels from all models
        if results["threat_analysis"] and "error" not in results["threat_analysis"]:
            risk = results["threat_analysis"].get("risk_level", "UNKNOWN")
            risk_levels.append(risk)
            if results["threat_analysis"].get("flagged"):
                recommendations.append(
                    f"Log analysis flagged: {risk} risk detected — "
                    f"investigate source immediately"
                )

        if results["code_analysis"] and "error" not in results["code_analysis"]:
            risk = results["code_analysis"].get("risk_level", "UNKNOWN")
            risk_levels.append(risk)
            recommendation = results["code_analysis"].get("recommendation")
            if recommendation:
                recommendations.append(f"Code analysis: {recommendation}")

        if results["urgency_analysis"] and "error" not in results["urgency_analysis"]:
            urgency = results["urgency_analysis"].get("urgency_level", "UNKNOWN")
            results["overall_urgency"] = urgency
            action = results["urgency_analysis"].get("recommended_action")
            if action:
                recommendations.append(f"Threat bulletin: {action}")

        # Determine overall risk — take the highest severity
        if risk_levels:
            results["overall_risk"] = min(
                risk_levels,
                key=lambda x: self.RISK_PRIORITY.get(x, 99)
            )

        results["recommendations"] = recommendations

        # Generate summary
        results["summary"] = self._generate_summary(results)

        return results

    def _generate_summary(self, results: dict) -> str:
        """
        Generates a human readable summary of the threat assessment.
        """
        overall_risk = results.get("overall_risk", "UNKNOWN")
        overall_urgency = results.get("overall_urgency", "UNKNOWN")
        models_count = len(results.get("models_used", []))
        timestamp = results.get("timestamp", "")

        summary_parts = [
            f"CyberSentinel Assessment — {timestamp}",
            f"Models Run: {models_count}",
            f"Overall Risk Level: {overall_risk}",
            f"Overall Urgency: {overall_urgency}",
        ]

        # Add threat specific details
        if results["threat_analysis"] and "error" not in results["threat_analysis"]:
            threat = results["threat_analysis"]
            summary_parts.append(
                f"Log Analysis: {threat.get('risk_level')} risk "
                f"(confidence: {threat.get('confidence')})"
            )

        if results["code_analysis"] and "error" not in results["code_analysis"]:
            code = results["code_analysis"]
            summary_parts.append(
                f"Code Analysis: {code.get('threat_type')} detected — "
                f"{code.get('risk_level')} risk"
            )

        if results["urgency_analysis"] and "error" not in results["urgency_analysis"]:
            urgency = results["urgency_analysis"]
            summary_parts.append(
                f"Bulletin Urgency: {urgency.get('urgency_level')} — "
                f"respond {urgency.get('response_time')}"
            )

        return " | ".join(summary_parts)


# Single instance used across the application
pipeline = CyberSentinelPipeline()