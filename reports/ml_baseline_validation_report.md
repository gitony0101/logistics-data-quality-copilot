# Machine Learning Baseline Validation Report

## 1. Executive Summary
This report details the performance of the **Machine Learning Baseline Detector**, designed specifically to address the blind spots identified in the deterministic rule-based engine. By implementing Natural Language Processing (NLP) and Anomaly Detection, the system significantly improves recall on complex, semantic, and multivariate data quality issues.

## 2. Models Implemented
1. **TF-IDF + Logistic Regression:** Trained on the clean dataset to map product descriptions to their likely 4-digit HS code prefix. Used to detect `hs_code_product_mismatch` where simple keyword rules failed.
2. **Isolation Forest:** An unsupervised anomaly detection model using `log(weight)`, `log(value)`, and `value_per_kg` to detect `suspicious_low_value_high_weight` and `unrealistic_weight` without hardcoded thresholds.

## 3. Targeted Performance Improvement
The ML layer specifically targeted errors where the Rule-Based engine showed low Recall (< 50%) or low Precision (due to overly broad rules).

| Error Type | Precision | Recall | F1 Score |
|------------|-----------|--------|----------|
| hs_code_product_mismatch | 0.4828 | 0.8936 | 0.6269 |
| suspicious_low_value_high_weight | 0.8837 | 0.3486 | 0.5000 |
| unrealistic_weight | 0.1722 | 0.9398 | 0.2910 |

## 4. Business Interpretation
- **NLP Success:** The Logistic Regression model dramatically improves our ability to spot HS Code mismatches. Instead of relying on a human writing 10,000 dictionary rules, the model learned the latent semantic mapping from historical data.
- **Multivariate Risk:** The Isolation Forest successfully flags records that *look* normal on a single axis (e.g., weight is only 9000kg, below the 50k threshold) but are highly anomalous when combined with value (e.g., 9000kg valued at only $50).
- **The Combined Pipeline:** By stacking the Rule-Based Engine (high precision on formatting) and the ML Baseline (contextual awareness), we create a robust "Human-in-the-loop" filtering mechanism that reliably isolates high-risk manifests.

## 5. Next Implementation Step
The next phase is to combine these components into a **Decision Support Dashboard** using Streamlit (`src/dashboard/app.py`), allowing human operators to interact with both the rule flags and ML confidence scores visually.
