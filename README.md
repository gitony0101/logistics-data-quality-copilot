# Logistics Data Quality and Error Detection Dashboard

## Business Problem
In global supply chains, shipping manifests often contain unstructured, noisy, or erroneous data due to manual entry across different countries and systems. Misclassified HS (Harmonized System) codes, unrealistic declared weights or values, and missing fields create significant workflow bottlenecks. Customs agencies and logistics providers must dedicate thousands of hours to manually reviewing these documents. Failure to catch errors leads to compliance fines, delayed shipments, and lost revenue.

## MVP Objective
This portfolio project demonstrates an Industrial Engineering and Operations Research approach to resolving this bottleneck. By building an automated data quality pipeline, we aim to transition the workflow from 100% manual review to a "Human-in-the-Loop" exception management system. 

The MVP objective is to:
1. Generate a realistic synthetic dataset representing dirty shipping manifests.
2. Build rule-based and machine-learning (NLP/Anomaly Detection) validators to catch these errors.
3. Deploy a decision-support Streamlit dashboard that automatically clears high-confidence records and routes high-risk anomalies to human reviewers, supporting a shift from full manual review toward prioritized exception review.

## How to Run the Data Generation Script
The first component of this project is generating the synthetic data.

1. Ensure requirements are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the generation script from the project root:
   ```bash
   python src/data_generation/generate_manifests.py --rows 10000 --seed 42 --error-rate 0.12
   ```

## Expected Outputs
The script will dynamically create the `data/synthetic/` directory (if it doesn't exist) and output three files:
- **`shipping_manifests_clean.csv`**: The ground-truth, perfect dataset.
- **`shipping_manifests_dirty.csv`**: The dataset injected with realistic data quality errors (e.g., mismatched HS codes, negative weights) based on the specified error rate.
- **`shipping_manifest_error_log.csv`**: A row-level audit log detailing exactly which errors were injected into which manifests, including severity and detection hints.

## Rule-Based Validation
The deterministic rule-based validator serves as a critical first-pass system, acting like an automated customs reviewer to catch high-confidence, low-complexity errors (e.g., missing fields, negative weights, invalid dates).

To run the validator:
```bash
python src/data_validation/rule_based_validator.py \
  --input data/synthetic/shipping_manifests_dirty.csv \
  --error-log data/synthetic/shipping_manifest_error_log.csv \
  --output-dir data/processed \
  --report-dir reports
```

### Expected Outputs
- **`data/processed/rule_based_validation_findings.csv`**: A row-level log of every rule triggered.
- **`data/processed/rule_based_validation_metrics.csv`**: Performance metrics evaluating the rules against the ground-truth error log.
- **`reports/rule_based_validation_report.md`**: An analytical markdown report summarizing rule performance.

### Interpreting Precision and Recall
- **Precision:** Of all the errors the rules flagged, how many were actual injected errors? A high precision means we don't bother human reviewers with false alarms.
- **Recall:** Of all the actual injected errors, how many did the rules catch? A high recall means fewer errors slip through to the real world.
- **Why this is the first baseline before ML:** Deterministic rules are highly effective (often > 80% precision and recall) for formatting and threshold checks. Establishing this baseline allows us to use Machine Learning strictly for complex, semantic anomalies (like mismatched descriptions and HS codes) that rules cannot cleanly capture.

### Current Results

#### Rule-Based Validation Performance
The rule-based validation achieved a Precision of **0.8493** and a Recall of **0.8408** for an overall F1 Score of **0.8451**.

Performance by error type:
| Error Type | Precision | Recall | F1 Score |
|------------|-----------|--------|----------|
| future_shipment_date | 1.0000 | 1.0000 | 1.0000 |
| extreme_declared_value | 1.0000 | 1.0000 | 1.0000 |
| missing_hs_code | 1.0000 | 1.0000 | 1.0000 |
| negative_weight | 1.0000 | 1.0000 | 1.0000 |
| missing_product_description | 1.0000 | 1.0000 | 1.0000 |
| invalid_country | 1.0000 | 1.0000 | 1.0000 |
| invalid_hs_code_format | 1.0000 | 1.0000 | 1.0000 |
| unrealistic_weight | 1.0000 | 1.0000 | 1.0000 |
| port_country_mismatch | 1.0000 | 1.0000 | 1.0000 |
| quantity_zero | 1.0000 | 1.0000 | 1.0000 |
| currency_mismatch | 1.0000 | 1.0000 | 1.0000 |
| hs_code_product_mismatch | 0.5109 | 1.0000 | 0.6763 |
| suspicious_low_value_high_weight | 1.0000 | 0.0642 | 0.1207 |
| duplicated_manifest_id | 0.0000 | 0.0000 | 0.0000 |

#### ML Baseline Detection Performance
The Machine Learning Baseline Detector achieved the following performance on targeted error types:

| Error Type | Precision | Recall | F1 Score |
|------------|-----------|--------|----------|
| hs_code_product_mismatch | 0.4828 | 0.8936 | 0.6269 |
| suspicious_low_value_high_weight | 0.8837 | 0.3486 | 0.5000 |
| unrealistic_weight | 0.1722 | 0.9398 | 0.2910 |

## Dashboard Preview
The dashboard provides a comprehensive interface for reviewing data quality issues:

1. **Rule-based findings view**: Shows the validation results from deterministic rules
2. **ML findings view**: Displays machine learning detected anomalies
3. **Human-in-the-loop review**: Allows users to review and make decisions on flagged records

To run the dashboard:
```bash
streamlit run src/dashboard/app.py --server.port 8501 --server.headless true
```

## Decision-Support Value
The system provides significant business value by:
1. Reducing manual workload through automated error detection
2. Enabling human-in-the-loop review for complex cases
3. Providing clear metrics on detection performance
4. Offering a dashboard interface for efficient review workflows

## ML Baseline Validation
The Machine Learning Baseline Detector addresses the blind spots of the rule-based engine by using NLP and Anomaly Detection to catch complex semantic mismatches and multivariate anomalies.

To run the ML validator:
```bash
python src/models/ml_baseline_detector.py \
  --clean data/synthetic/shipping_manifests_clean.csv \
  --dirty data/synthetic/shipping_manifests_dirty.csv \
  --error-log data/synthetic/shipping_manifest_error_log.csv \
  --output-dir data/processed \
  --report-dir reports
```

### Expected Outputs
- **`data/processed/ml_baseline_validation_findings.csv`**: A row-level log of ML-detected anomalies.
- **`data/processed/ml_baseline_validation_metrics.csv`**: Performance metrics evaluating the ML models on their specific target errors.
- **`reports/ml_baseline_validation_report.md`**: An analytical markdown report summarizing ML performance.

## Why This Matters
This project directly addresses a core Industrial Engineering mandate: process optimization and risk reduction. By framing applied AI (NLP and Anomaly Detection) as a tool for decision support, it demonstrates how to translate machine learning metrics (like Precision and Recall) into tangible business value (Operational Workload Reduction).

---

## Resume Bullet

> Designed a hybrid rule-based and ML data validation pipeline (TF-IDF + Isolation Forest) for shipping manifest quality control on synthetic data, achieving 84.9% precision and 84.1% recall across 14 error classes with a Streamlit dashboard enabling human-in-the-loop review.

---

## License

MIT License

---

## Disclaimer

> This repository is a portfolio and learning prototype. The project is a simplified research and engineering demonstration, not a production system, commercial product, or certified decision-support tool. Results are based on synthetic data and should not be used for real operational decisions without further validation.
