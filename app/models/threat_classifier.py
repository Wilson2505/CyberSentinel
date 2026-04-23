from transformers import pipeline
from app.config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreatClassifier:
    """
    Model 1: Analyses text-based security data such as log files
    and threat reports to classify threat types and confidence levels.
    
    Uses a HuggingFace pre-trained NLP model fine-tuned for
    security-domain text classification.
    """
    
    def __init__(self):
        self.model_name = config.HF_THREAT_MODEL
        self.classifier = None
        self.is_loaded = False
        
    def load(self):
        """Load the model — called explicitly so we control when it loads."""
        try:
            logger.info(f"Loading threat classifier: {self.model_name}")
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                truncation=True,
                max_length=512
            )
            self.is_loaded = True
            logger.info("Threat classifier loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load threat classifier: {e}")
            raise
    
    def analyse(self, text: str) -> dict:
        """
        Analyse input text for threat indicators.
        
        Args:
            text: Raw log entry, security report, or threat description
            
        Returns:
            dict containing:
                - label: classified threat sentiment
                - confidence: model confidence score (0-1)
                - risk_level: derived risk assessment
                - raw_text: original input (truncated)
                - model_used: model name for report documentation
        """
        if not self.is_loaded:
            self.load()
        
        if not text or not text.strip():
            return {
                "error": "Empty input provided",
                "risk_level": "UNKNOWN"
            }
        
        try:
            # Truncate very long inputs for display
            display_text = text[:200] + "..." if len(text) > 200 else text
            
            result = self.classifier(text)[0]
            
            confidence = result["score"]
            label = result["label"]
            
            # Map model output to risk levels
            risk_level = self._map_to_risk_level(label, confidence)
            
            return {
                "label": label,
                "confidence": round(confidence, 4),
                "risk_level": risk_level,
                "raw_text": display_text,
                "model_used": self.model_name,
                "flagged": confidence >= config.CONFIDENCE_THRESHOLD
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "error": str(e),
                "risk_level": "UNKNOWN",
                "model_used": self.model_name
            }
    
    def _map_to_risk_level(self, label: str, confidence: float) -> str:
        """
        Maps model output labels to cybersecurity risk levels.
        This mapping will be refined during evaluation phase.
        """
        # NEGATIVE sentiment in security context = potentially malicious
        if label == "NEGATIVE":
            if confidence >= 0.90:
                return "CRITICAL"
            elif confidence >= 0.75:
                return "HIGH"
            elif confidence >= 0.60:
                return "MEDIUM"
            else:
                return "LOW"
        else:
            if confidence >= 0.90:
                return "SAFE"
            else:
                return "LOW"
    
    def evaluate_model(self, test_cases: list) -> dict:
        """
        Evaluates model performance on test cases.
        Used for model selection documentation in the report.
        
        Args:
            test_cases: list of dicts with 'text' and 'expected_risk' keys
            
        Returns:
            dict with accuracy metrics
        """
        if not self.is_loaded:
            self.load()
            
        results = []
        correct = 0
        
        for case in test_cases:
            result = self.analyse(case["text"])
            expected = case["expected_risk"]
            actual = result.get("risk_level", "UNKNOWN")
            is_correct = actual == expected
            
            if is_correct:
                correct += 1
                
            results.append({
                "text": case["text"][:100],
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
threat_classifier = ThreatClassifier()