# Shipping Manifest Error Detection

**One‑Sentence Summary**: An AI‑augmented pipeline that validates and corrects shipping manifest data using rule‑based checks, NLP semantic matching, and multivariate anomaly detection, with a human‑in‑the‑loop dashboard for review.

## Business Problem
Customs agencies and logistics providers suffer costly delays and compliance risks due to inaccurate or incomplete shipping manifest data.

## Target User
Customs officials, freight forwarders, and data‑quality analysts.

## Decision Supported
Whether a manifest can be automatically cleared, requires manual correction, or should be escalated for investigation.

## Input Data
- Synthetic or real manifest CSVs containing fields: `manifest_id`, `shipment_id`, `product_description`, `hs_code`, `declared_weight_kg`, `declared_value_usd`, etc.
- Ground‑truth error log CSV for evaluation.

## Methodology
1. **Rule‑based validator** – deterministic checks (negative weight, missing HS code, etc.).
2. **NLP model** – TF‑IDF + Logistic Regression to detect `hs_code_product_mismatch`.
3. **Anomaly detection** – Isolation Forest on log‑transformed weight/value features.
4. **Dashboard** – Streamlit UI for filtering, reviewing, and approving findings.

## MVP Scope
- End‑to‑end data ingestion, validation, ML detection, and dashboard.
- Generate reports (`rule_based_validation_report.md`, `ml_baseline_validation_report.md`).

## Standard Version
- Expand rule set, add active learning loop for NLP, and integrate with a database.

## Research Grade Extension
- Incorporate transformer‑based text classification, develop a semi‑supervised labeling pipeline, and evaluate on real customs data.

## Portfolio Deliverables
- Source code repository with Python scripts and Streamlit app.
- Markdown reports with performance metrics.
- **Resume Bullet**: "Designed a hybrid rule-based and ML data validation pipeline (TF-IDF + Isolation Forest) for shipping manifest quality control on synthetic data, achieving 84.9% precision and 84.1% recall across 14 error classes with a Streamlit dashboard enabling human-in-the-loop review."

## Resume Bullet Draft
- Designed and implemented a hybrid rule‑based and ML pipeline (TF‑IDF + Isolation Forest) achieving 0.85 F1 on synthetic data, with a Streamlit dashboard for human-in-the-loop decision support.
