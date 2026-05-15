# Methodology: Logistics Data Quality Pipeline

## Data Generation
To safely simulate proprietary shipping manifest data, we generated a synthetic dataset with realistic supply chain properties (e.g., origin/destination pairs, HS codes, weights, and values). We injected 14 specific error classes representing real-world operational issues to create a "dirty" dataset alongside a ground-truth error log.

## Rule-Based Validation Methodology
Before introducing any statistical modeling or machine learning, we built a deterministic rule engine. 
- **Purpose:** To act as a first-pass "filter" that catches obvious, formatting-based, or threshold-based anomalies (e.g., negative weights, missing HS codes, future dates).
- **Why deterministic rules first:** In an industrial engineering context, transparent rules are cheaper to maintain, highly interpretable, and offer perfect precision for formatting violations. It is inefficient to use a complex ML model to find an empty string.

## Limitations of Rules
While rule engines achieve high precision/recall on structured constraints, they fall short on semantic or multivariate anomalies. For example:
- Mapping "auto parts" to a specific HS code using string matching yields low recall, as the description could be phased in hundreds of ways not captured by the hardcoded dictionary.
- Hard threshold rules (e.g., weight > 50,000 kg) fail to catch items that are "suspiciously heavy" for their specific product category but under the absolute threshold.

## ML and NLP Implementation

After establishing the rule-based baseline, we introduced two machine learning components to address the blind spots identified above:

### 1. NLP Classification (TF-IDF + Logistic Regression)
We implemented a text classification pipeline using TF-IDF vectorization and Logistic Regression to handle fuzzy semantic matching between product descriptions and HS codes. This targets the `hs_code_product_mismatch` error class that rule-based checks cannot capture.

### 2. Anomaly Detection (Isolation Forest)
We implemented an Isolation Forest model to detect multivariate anomalies — suspicious combinations of weight, value, and product category — without relying on arbitrary hardcoded limits.

## Current ML Results
- **hs_code_product_mismatch**: 0.4828 precision, 0.8936 recall (F1=0.6269) — the NLP classifier trades lower precision for substantially higher recall (0.8936 vs 0.5109) compared to the rule-based approach (F1=0.6763), catching more true mismatches at the cost of more false positives.
- **suspicious_low_value_high_weight**: 0.8837 precision, 0.3486 recall (F1=0.5000) — high precision but limited recall.
- **unrealistic_weight**: 0.1722 precision, 0.9398 recall (F1=0.2910) — high recall but many false positives, suggesting further feature engineering is needed.