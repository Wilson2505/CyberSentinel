# CyberSentinel User Testing Protocol

## Final Year Project — CM3020 Artificial Intelligence

**Tester:** Wilson Tan Guan Hua
**Supervisor:** Prof Yip See Wai
**Target Participants:** 5 minimum, computing background preferred

---

## Objectives

1. Evaluate the usability of the CyberSentinel dashboard
2. Assess whether the system's risk assessments align with participant judgement
3. Identify interface improvements through structured feedback
4. Collect SUS scores for quantitative usability measurement

---

## Pre-Session Setup

Before each participant session ensure:

- ollama serve is running in Tab 1
- uvicorn app.main:app --reload --port 8000 is running in Tab 3
- Browser is open at http://localhost:8000
- All previous test analyses are cleared if needed

---

## Participant Instructions

Read this to each participant before they begin:

> "You are acting as a security analyst reviewing incoming security data.
> I will give you three scenarios to work through using this tool.
> Please think aloud as you use the system — tell me what you are
> looking at, what you expect to happen, and whether the results
> make sense to you. There are no right or wrong answers.
> I am testing the tool, not you."

---

## Test Scenarios

### Scenario 1: Confirmed Attack (High Severity)

**Log Text input:**
