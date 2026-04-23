"""
Model 1 Evaluation: DistilBERT Threat Classifier
=================================================
This script evaluates the performance of distilbert-base-uncased-finetuned-sst-2-english
on cybersecurity text classification tasks.

Purpose: To document model selection decision for the PPR report.
Date: April 2026
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.threat_classifier import ThreatClassifier
import json
from datetime import datetime

# ── Test Cases ──────────────────────────────────────────────────────────────

SUSPICIOUS_CASES = [
    {
        "text": "Failed login attempt for user admin from IP 192.168.1.105 - attempt 5 of 5",
        "expected_risk": "HIGH",
        "category": "Brute Force"
    },
    {
        "text": "Unauthorized access detected on port 22 from external IP 45.33.32.156",
        "expected_risk": "HIGH",
        "category": "Unauthorized Access"
    },
    {
        "text": "SQL injection attempt detected in web request: SELECT * FROM users WHERE id=1 OR 1=1",
        "expected_risk": "CRITICAL",
        "category": "SQL Injection"
    },
    {
        "text": "Malware signature detected in file upload: trojan.genericKD.46678",
        "expected_risk": "CRITICAL",
        "category": "Malware"
    },
    {
        "text": "Port scan detected from IP 203.0.113.42 targeting ports 22,80,443,8080",
        "expected_risk": "HIGH",
        "category": "Port Scan"
    },
    {
        "text": "Ransomware encryption process detected on multiple files in /home/user/documents",
        "expected_risk": "CRITICAL",
        "category": "Ransomware"
    },
    {
        "text": "Phishing email detected containing malicious link to fake banking portal",
        "expected_risk": "HIGH",
        "category": "Phishing"
    },
    {
        "text": "Data exfiltration attempt detected - large outbound transfer to unknown IP",
        "expected_risk": "CRITICAL",
        "category": "Data Exfiltration"
    }
]

BENIGN_CASES = [
    {
        "text": "User successfully logged in from trusted IP 10.0.0.1",
        "expected_risk": "SAFE",
        "category": "Normal Login"
    },
    {
        "text": "System backup completed successfully at 02:00 UTC",
        "expected_risk": "SAFE",
        "category": "Scheduled Task"
    },
    {
        "text": "SSL certificate renewed successfully for domain example.com",
        "expected_risk": "SAFE",
        "category": "Maintenance"
    },
    {
        "text": "Database backup successful - 2.3GB archived to secure storage",
        "expected_risk": "SAFE",
        "category": "Backup"
    },
    {
        "text": "System health check passed - all services running normally",
        "expected_risk": "SAFE",
        "category": "Health Check"
    },
    {
        "text": "User password changed successfully following security policy",
        "expected_risk": "SAFE",
        "category": "Policy Compliance"
    },
    {
        "text": "Scheduled vulnerability scan completed - no critical issues found",
        "expected_risk": "SAFE",
        "category": "Security Scan"
    },
    {
        "text": "Firewall rules updated successfully by administrator",
        "expected_risk": "SAFE",
        "category": "Admin Action"
    }
]

# ── Evaluation Functions ─────────────────────────────────────────────────────

def run_evaluation():
    print("=" * 70)
    print("CyberSentinel — Model 1 Evaluation Report")
    print("Model: distilbert-base-uncased-finetuned-sst-2-english")
    print(f"Date:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    clf = ThreatClassifier()
    clf.load()

    print("\n📋 Loading model... done\n")

    # ── Suspicious Cases ────────────────────────────────────────────────────
    print("─" * 70)
    print("SECTION 1: SUSPICIOUS / MALICIOUS LOG ENTRIES")
    print("─" * 70)

    suspicious_results = []
    suspicious_flagged = 0

    for i, case in enumerate(SUSPICIOUS_CASES, 1):
        result = clf.analyse(case["text"])
        flagged = result.get("flagged", False)
        if flagged:
            suspicious_flagged += 1

        status = "✓ FLAGGED" if flagged else "✗ MISSED"

        print(f"\n[{i}] Category: {case['category']}")
        print(f"    Input:      {case['text'][:80]}...")
        print(f"    Label:      {result.get('label')}")
        print(f"    Confidence: {result.get('confidence')}")
        print(f"    Risk Level: {result.get('risk_level')}")
        print(f"    Status:     {status}")

        suspicious_results.append({
            "category": case["category"],
            "label": result.get("label"),
            "confidence": result.get("confidence"),
            "risk_level": result.get("risk_level"),
            "flagged": flagged
        })

    # ── Benign Cases ────────────────────────────────────────────────────────
    print("\n")
    print("─" * 70)
    print("SECTION 2: BENIGN / NORMAL LOG ENTRIES")
    print("─" * 70)

    benign_results = []
    benign_correct = 0

    for i, case in enumerate(BENIGN_CASES, 1):
        result = clf.analyse(case["text"])
        flagged = result.get("flagged", False)
        
        # For benign cases, correct = NOT flagged
        if not flagged:
            benign_correct += 1

        status = "✓ CORRECTLY SAFE" if not flagged else "✗ FALSE POSITIVE"

        print(f"\n[{i}] Category: {case['category']}")
        print(f"    Input:      {case['text'][:80]}")
        print(f"    Label:      {result.get('label')}")
        print(f"    Confidence: {result.get('confidence')}")
        print(f"    Risk Level: {result.get('risk_level')}")
        print(f"    Status:     {status}")

        benign_results.append({
            "category": case["category"],
            "label": result.get("label"),
            "confidence": result.get("confidence"),
            "risk_level": result.get("risk_level"),
            "flagged": flagged
        })

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n")
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    total_suspicious = len(SUSPICIOUS_CASES)
    total_benign = len(BENIGN_CASES)
    total_cases = total_suspicious + total_benign

    # Detection rate — how many suspicious cases were correctly flagged
    detection_rate = suspicious_flagged / total_suspicious
    
    # Specificity — how many benign cases were correctly NOT flagged
    specificity = benign_correct / total_benign
    
    # Overall accuracy
    overall_correct = suspicious_flagged + benign_correct
    overall_accuracy = overall_correct / total_cases

    print(f"\n  Model:               distilbert-base-uncased-finetuned-sst-2-english")
    print(f"  Total Test Cases:    {total_cases}")
    print(f"")
    print(f"  Suspicious Cases:    {total_suspicious}")
    print(f"  Correctly Flagged:   {suspicious_flagged}/{total_suspicious}")
    print(f"  Detection Rate:      {detection_rate:.1%}")
    print(f"")
    print(f"  Benign Cases:        {total_benign}")
    print(f"  Correctly Safe:      {benign_correct}/{total_benign}")
    print(f"  Specificity:         {specificity:.1%}")
    print(f"")
    print(f"  Overall Accuracy:    {overall_accuracy:.1%}")
    print(f"")

    # ── Interpretation ───────────────────────────────────────────────────────
    print("─" * 70)
    print("INTERPRETATION FOR REPORT")
    print("─" * 70)
    print(f"""
This evaluation tests whether a general-purpose sentiment analysis model
(DistilBERT fine-tuned on SST-2) can serve as a proxy threat detector
for cybersecurity log classification.

Key Findings:
  - Detection Rate of {detection_rate:.1%} means the model correctly identified
    {suspicious_flagged} out of {total_suspicious} malicious log entries as negative/threatening.

  - Specificity of {specificity:.1%} means the model correctly identified
    {benign_correct} out of {total_benign} benign entries as non-threatening.

  - Overall Accuracy: {overall_accuracy:.1%} across all {total_cases} test cases.

Limitation Identified:
  DistilBERT was trained on movie review sentiment, not cybersecurity
  text. This means it detects threatening LANGUAGE but may miss
  technically suspicious but neutrally-worded log entries.

Model Selection Decision:
  {"ACCEPTED as baseline — performance is sufficient for prototype stage." 
   if overall_accuracy >= 0.6 
   else "REJECTED — insufficient accuracy for cybersecurity use case."}
  
  For production: recommend replacing with a security-domain specific
  model such as SecRoBERTa or a fine-tuned CyberBERT model.
  This comparison will be documented in the PPR model selection section.
    """)

    # ── Save results ─────────────────────────────────────────────────────────
    output = {
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "evaluation_date": datetime.now().isoformat(),
        "metrics": {
            "total_cases": total_cases,
            "detection_rate": round(detection_rate, 4),
            "specificity": round(specificity, 4),
            "overall_accuracy": round(overall_accuracy, 4)
        },
        "suspicious_results": suspicious_results,
        "benign_results": benign_results
    }

    output_path = "docs/model1_evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()