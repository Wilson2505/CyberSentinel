"""
Model 1 Evaluation: SecRoBERTa vs DistilBERT Comparison
========================================================
This script evaluates SecRoBERTa against the DistilBERT baseline
to document the model improvement for the final report.

Purpose: Evidence-based model selection for final submission
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import pipeline
from datetime import datetime
import json

SUSPICIOUS_CASES = [
    {"text": "Failed login attempt for user admin from IP 192.168.1.105 - attempt 5 of 5", "category": "Brute Force"},
    {"text": "Unauthorized access detected on port 22 from external IP 45.33.32.156", "category": "Unauthorized Access"},
    {"text": "SQL injection attempt detected in web request: SELECT * FROM users WHERE id=1 OR 1=1", "category": "SQL Injection"},
    {"text": "Malware signature detected in file upload: trojan.genericKD.46678", "category": "Malware"},
    {"text": "Port scan detected from IP 203.0.113.42 targeting ports 22,80,443,8080", "category": "Port Scan"},
    {"text": "Ransomware encryption process detected on multiple files in /home/user/documents", "category": "Ransomware"},
    {"text": "Phishing email detected containing malicious link to fake banking portal", "category": "Phishing"},
    {"text": "Data exfiltration attempt detected - large outbound transfer to unknown IP", "category": "Data Exfiltration"}
]

BENIGN_CASES = [
    {"text": "User successfully logged in from trusted IP 10.0.0.1", "category": "Normal Login"},
    {"text": "System backup completed successfully at 02:00 UTC", "category": "Scheduled Task"},
    {"text": "SSL certificate renewed successfully for domain example.com", "category": "Maintenance"},
    {"text": "Database backup successful - 2.3GB archived to secure storage", "category": "Backup"},
    {"text": "System health check passed - all services running normally", "category": "Health Check"},
    {"text": "User password changed successfully following security policy", "category": "Policy Compliance"},
    {"text": "Scheduled vulnerability scan completed - no critical issues found", "category": "Security Scan"},
    {"text": "Firewall rules updated successfully by administrator", "category": "Admin Action"}
]

def evaluate_model(model_name, classifier):
    print(f"\nEvaluating: {model_name}")
    print("-" * 50)

    suspicious_flagged = 0
    benign_correct = 0
    results = {"suspicious": [], "benign": []}

    for case in SUSPICIOUS_CASES:
        result = classifier(case["text"])[0]
        label = result["label"]
        confidence = round(result["score"], 4)

        # Flag if negative sentiment above threshold
        flagged = label in ["NEGATIVE", "LABEL_1"] and confidence >= 0.7

        if flagged:
            suspicious_flagged += 1

        results["suspicious"].append({
            "category": case["category"],
            "label": label,
            "confidence": confidence,
            "flagged": flagged
        })

        status = "FLAGGED" if flagged else "MISSED"
        print(f"  [{status}] {case['category']}: {label} ({confidence})")

    print()

    for case in BENIGN_CASES:
        result = classifier(case["text"])[0]
        label = result["label"]
        confidence = round(result["score"], 4)

        flagged = label in ["NEGATIVE", "LABEL_1"] and confidence >= 0.7

        if not flagged:
            benign_correct += 1

        results["benign"].append({
            "category": case["category"],
            "label": label,
            "confidence": confidence,
            "flagged": flagged
        })

        status = "SAFE" if not flagged else "FALSE POSITIVE"
        print(f"  [{status}] {case['category']}: {label} ({confidence})")

    total = len(SUSPICIOUS_CASES) + len(BENIGN_CASES)
    detection_rate = suspicious_flagged / len(SUSPICIOUS_CASES)
    specificity = benign_correct / len(BENIGN_CASES)
    accuracy = (suspicious_flagged + benign_correct) / total

    print(f"\n  Detection Rate: {detection_rate:.1%}")
    print(f"  Specificity:    {specificity:.1%}")
    print(f"  Overall Accuracy: {accuracy:.1%}")

    return {
        "model": model_name,
        "detection_rate": round(detection_rate, 4),
        "specificity": round(specificity, 4),
        "overall_accuracy": round(accuracy, 4),
        "suspicious_flagged": suspicious_flagged,
        "benign_correct": benign_correct,
        "results": results
    }

def run_comparison():
    print("=" * 60)
    print("CyberSentinel Model 1 Comparison")
    print("DistilBERT SST-2 vs SecRoBERTa")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Evaluate DistilBERT baseline
    print("\nLoading DistilBERT SST-2...")
    distilbert = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512
    )
    distilbert_results = evaluate_model(
        "distilbert-base-uncased-finetuned-sst-2-english",
        distilbert
    )

    # Evaluate SecRoBERTa
    print("\nLoading SecRoBERTa...")
    try:
        secroberta = pipeline(
            "text-classification",
            model="jackaduma/SecRoBERTa",
            truncation=True,
            max_length=512
        )
        secroberta_results = evaluate_model(
            "jackaduma/SecRoBERTa",
            secroberta
        )
        secroberta_available = True
    except Exception as e:
        print(f"SecRoBERTa failed to load: {e}")
        secroberta_results = None
        secroberta_available = False

    # Print comparison
    print("\n")
    print("=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'DistilBERT':>12} {'SecRoBERTa':>12}")
    print("-" * 50)

    if secroberta_available:
        print(f"{'Detection Rate':<25} {distilbert_results['detection_rate']:.1%}{'':<6} {secroberta_results['detection_rate']:.1%}")
        print(f"{'Specificity':<25} {distilbert_results['specificity']:.1%}{'':<6} {secroberta_results['specificity']:.1%}")
        print(f"{'Overall Accuracy':<25} {distilbert_results['overall_accuracy']:.1%}{'':<6} {secroberta_results['overall_accuracy']:.1%}")

        winner = "SecRoBERTa" if secroberta_results["overall_accuracy"] > distilbert_results["overall_accuracy"] else "DistilBERT"
        print(f"\nRecommended Model: {winner}")

    # Save results
    output = {
        "evaluation_date": datetime.now().isoformat(),
        "distilbert": distilbert_results,
        "secroberta": secroberta_results
    }

    with open("docs/model1_comparison_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to docs/model1_comparison_results.json")
    print("=" * 60)

if __name__ == "__main__":
    run_comparison()