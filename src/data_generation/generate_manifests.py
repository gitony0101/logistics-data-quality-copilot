import pandas as pd
import numpy as np
from faker import Faker
import argparse
from pathlib import Path
import random
import datetime

def setup_argparse() -> argparse.ArgumentParser:
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic shipping manifests.")
    parser.add_argument("--rows", type=int, default=10000, help="Number of rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--error-rate", type=float, default=0.10, help="Proportion of rows to inject errors into (0.0 to 1.0).")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Directory to save output files.")
    return parser

# Define product categories and realistic HS code prefixes
PRODUCT_MAPPINGS = {
    "auto parts": {"hs_prefix": "8708", "unit": "kg", "weight_range": (50, 5000), "val_per_kg_range": (5, 50)},
    "frozen seafood": {"hs_prefix": "0303", "unit": "kg", "weight_range": (500, 20000), "val_per_kg_range": (8, 25)},
    "machinery components": {"hs_prefix": "8431", "unit": "kg", "weight_range": (100, 10000), "val_per_kg_range": (10, 100)},
    "electronics": {"hs_prefix": "8542", "unit": "kg", "weight_range": (10, 2000), "val_per_kg_range": (50, 500)},
    "textiles": {"hs_prefix": "6302", "unit": "kg", "weight_range": (100, 5000), "val_per_kg_range": (2, 15)},
    "furniture": {"hs_prefix": "9403", "unit": "kg", "weight_range": (200, 8000), "val_per_kg_range": (3, 20)},
    "agricultural equipment": {"hs_prefix": "8432", "unit": "kg", "weight_range": (500, 15000), "val_per_kg_range": (8, 40)},
    "plastic packaging": {"hs_prefix": "3923", "unit": "kg", "weight_range": (500, 10000), "val_per_kg_range": (1, 5)},
    "construction materials": {"hs_prefix": "6802", "unit": "kg", "weight_range": (1000, 25000), "val_per_kg_range": (0.5, 3)},
    "medical supplies": {"hs_prefix": "9018", "unit": "kg", "weight_range": (20, 1500), "val_per_kg_range": (20, 200)},
}

PORTS = {
    "US": ["Port of Los Angeles", "Port of Long Beach", "Port of New York and New Jersey", "Port of Savannah", "Port of Houston"],
    "CN": ["Port of Shanghai", "Port of Shenzhen", "Port of Ningbo-Zhoushan", "Port of Guangzhou", "Port of Qingdao"],
    "CA": ["Port of Vancouver", "Port of Montreal", "Port of Halifax", "Port of Prince Rupert"],
    "DE": ["Port of Hamburg", "Port of Bremen"],
    "NL": ["Port of Rotterdam", "Port of Amsterdam"]
}

COUNTRIES = list(PORTS.keys())

def generate_clean_manifests(num_rows: int, fake: Faker) -> pd.DataFrame:
    """Generate a clean, structurally sound synthetic dataset."""
    data = []
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=2*365) # Last 2 years

    for _ in range(num_rows):
        category = random.choice(list(PRODUCT_MAPPINGS.keys()))
        meta = PRODUCT_MAPPINGS[category]
        
        # Generate valid HS code
        hs_code = f"{meta['hs_prefix']}{random.randint(10, 99)}{random.randint(10, 99)}"
        
        # Generate realistic weights and values
        weight = round(random.uniform(*meta["weight_range"]), 2)
        val_per_kg = random.uniform(*meta["val_per_kg_range"])
        value = round(weight * val_per_kg, 2)
        
        orig_country = random.choice(COUNTRIES)
        dest_country = random.choice([c for c in COUNTRIES if c != orig_country])

        row = {
            "manifest_id": f"MAN-{fake.unique.random_number(digits=8, fix_len=True)}",
            "shipment_id": f"SHP-{fake.unique.random_number(digits=8, fix_len=True)}",
            "container_id": f"{fake.lexify(text='????')} {fake.numerify(text='#######')}",
            "shipment_date": fake.date_between(start_date=start_date, end_date=end_date).isoformat(),
            "origin_country": orig_country,
            "destination_country": dest_country,
            "origin_port": random.choice(PORTS[orig_country]),
            "destination_port": random.choice(PORTS[dest_country]),
            "carrier_name": fake.company() + " Shipping",
            "importer_name": fake.company(),
            "exporter_name": fake.company(),
            "product_category": category, # Helper column, will be used to generate description
            "product_description": f"{fake.word().capitalize()} {category} - {fake.catch_phrase()}",
            "hs_code": hs_code,
            "declared_weight_kg": weight,
            "declared_value_usd": value,
            "quantity": random.randint(1, 1000),
            "unit": meta["unit"],
            "currency": "USD",
            "transport_mode": random.choice(["Ocean", "Air", "Rail", "Truck"]),
            "risk_category": "low", # Base default
            "expected_review_status": "Auto Approve"
        }
        
        # Basic business rules for risk
        if value > 500000 or category in ["electronics", "medical supplies"]:
            row["risk_category"] = "medium"
            if value > 1000000:
                row["risk_category"] = "high"
                row["expected_review_status"] = "High Risk Review"
                
        data.append(row)
        
    df = pd.DataFrame(data)
    # Drop the helper column
    df = df.drop(columns=['product_category'])
    return df

def inject_errors(df: pd.DataFrame, error_rate: float, fake: Faker) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject realistic errors into a copy of the dataset and generate an error log."""
    dirty_df = df.copy()
    num_errors_to_inject = int(len(df) * error_rate)
    error_indices = random.sample(range(len(df)), num_errors_to_inject)
    
    error_log = []
    
    error_types = [
        "missing_hs_code", "invalid_hs_code_format", "hs_code_product_mismatch",
        "negative_weight", "unrealistic_weight", "missing_product_description",
        "duplicated_manifest_id", "invalid_country", "future_shipment_date",
        "currency_mismatch", "extreme_declared_value", "quantity_zero",
        "suspicious_low_value_high_weight", "port_country_mismatch"
    ]

    for idx in error_indices:
        err_type = random.choice(error_types)
        row = dirty_df.iloc[idx]
        log_entry = {
            "manifest_id": row["manifest_id"],
            "shipment_id": row["shipment_id"],
            "error_type": err_type,
            "affected_field": "",
            "original_value": "",
            "corrupted_value": "",
            "severity": "",
            "detection_hint": ""
        }

        if err_type == "missing_hs_code":
            log_entry["affected_field"] = "hs_code"
            log_entry["original_value"] = row["hs_code"]
            dirty_df.at[idx, "hs_code"] = np.nan
            log_entry["corrupted_value"] = "NaN"
            log_entry["severity"] = "high"
            log_entry["detection_hint"] = "Null check"

        elif err_type == "invalid_hs_code_format":
            log_entry["affected_field"] = "hs_code"
            log_entry["original_value"] = row["hs_code"]
            dirty_df.at[idx, "hs_code"] = f"{fake.lexify('??')}{row['hs_code'][2:]}" # Letters in HS code
            log_entry["corrupted_value"] = dirty_df.at[idx, "hs_code"]
            log_entry["severity"] = "medium"
            log_entry["detection_hint"] = "Regex format check (digits only)"

        elif err_type == "hs_code_product_mismatch":
            log_entry["affected_field"] = "hs_code"
            log_entry["original_value"] = row["hs_code"]
            dirty_df.at[idx, "hs_code"] = "01012100" # Purebred breeding horses (likely mismatch for electronics/auto parts)
            log_entry["corrupted_value"] = dirty_df.at[idx, "hs_code"]
            log_entry["severity"] = "high"
            log_entry["detection_hint"] = "NLP description to HS code mismatch"

        elif err_type == "negative_weight":
            log_entry["affected_field"] = "declared_weight_kg"
            log_entry["original_value"] = str(row["declared_weight_kg"])
            dirty_df.at[idx, "declared_weight_kg"] = -abs(row["declared_weight_kg"])
            log_entry["corrupted_value"] = str(dirty_df.at[idx, "declared_weight_kg"])
            log_entry["severity"] = "critical"
            log_entry["detection_hint"] = "Value < 0"

        elif err_type == "unrealistic_weight":
            log_entry["affected_field"] = "declared_weight_kg"
            log_entry["original_value"] = str(row["declared_weight_kg"])
            dirty_df.at[idx, "declared_weight_kg"] = row["declared_weight_kg"] * 1000 # E.g., typo adding zeros
            log_entry["corrupted_value"] = str(dirty_df.at[idx, "declared_weight_kg"])
            log_entry["severity"] = "medium"
            log_entry["detection_hint"] = "Anomaly detection (Isolation Forest)"

        elif err_type == "missing_product_description":
            log_entry["affected_field"] = "product_description"
            log_entry["original_value"] = row["product_description"]
            dirty_df.at[idx, "product_description"] = ""
            log_entry["corrupted_value"] = ""
            log_entry["severity"] = "high"
            log_entry["detection_hint"] = "Empty string check"

        elif err_type == "duplicated_manifest_id":
            log_entry["affected_field"] = "manifest_id"
            log_entry["original_value"] = row["manifest_id"]
            # Copy ID from another random row
            other_idx = random.choice([i for i in range(len(df)) if i != idx])
            dirty_df.at[idx, "manifest_id"] = dirty_df.at[other_idx, "manifest_id"]
            log_entry["corrupted_value"] = dirty_df.at[idx, "manifest_id"]
            log_entry["severity"] = "critical"
            log_entry["detection_hint"] = "Uniqueness constraint"

        elif err_type == "invalid_country":
            log_entry["affected_field"] = "origin_country"
            log_entry["original_value"] = row["origin_country"]
            dirty_df.at[idx, "origin_country"] = "ZZ" # Invalid ISO code
            log_entry["corrupted_value"] = "ZZ"
            log_entry["severity"] = "low"
            log_entry["detection_hint"] = "Lookup table validation"

        elif err_type == "future_shipment_date":
            log_entry["affected_field"] = "shipment_date"
            log_entry["original_value"] = row["shipment_date"]
            future_date = datetime.date.today() + datetime.timedelta(days=365)
            dirty_df.at[idx, "shipment_date"] = future_date.isoformat()
            log_entry["corrupted_value"] = dirty_df.at[idx, "shipment_date"]
            log_entry["severity"] = "medium"
            log_entry["detection_hint"] = "Date > today"

        elif err_type == "currency_mismatch":
            log_entry["affected_field"] = "currency"
            log_entry["original_value"] = row["currency"]
            dirty_df.at[idx, "currency"] = "EUR" # While value might be calculated as USD
            log_entry["corrupted_value"] = "EUR"
            log_entry["severity"] = "low"
            log_entry["detection_hint"] = "Rule: value expectations based on currency"
            
        elif err_type == "extreme_declared_value":
            log_entry["affected_field"] = "declared_value_usd"
            log_entry["original_value"] = str(row["declared_value_usd"])
            dirty_df.at[idx, "declared_value_usd"] = row["declared_value_usd"] * 10000
            log_entry["corrupted_value"] = str(dirty_df.at[idx, "declared_value_usd"])
            log_entry["severity"] = "high"
            log_entry["detection_hint"] = "Anomaly detection / Outlier"

        elif err_type == "quantity_zero":
            log_entry["affected_field"] = "quantity"
            log_entry["original_value"] = str(row["quantity"])
            dirty_df.at[idx, "quantity"] = 0
            log_entry["corrupted_value"] = "0"
            log_entry["severity"] = "medium"
            log_entry["detection_hint"] = "Value <= 0"

        elif err_type == "suspicious_low_value_high_weight":
             log_entry["affected_field"] = "declared_value_usd"
             log_entry["original_value"] = str(row["declared_value_usd"])
             dirty_df.at[idx, "declared_value_usd"] = 1.0 # 1 USD for potentially tons of weight
             log_entry["corrupted_value"] = "1.0"
             log_entry["severity"] = "high"
             log_entry["detection_hint"] = "Value per kg ratio anomaly"
             
        elif err_type == "port_country_mismatch":
             log_entry["affected_field"] = "origin_port"
             log_entry["original_value"] = row["origin_port"]
             # Pick a port from a different country
             wrong_country = random.choice([c for c in COUNTRIES if c != row["origin_country"]])
             dirty_df.at[idx, "origin_port"] = random.choice(PORTS[wrong_country])
             log_entry["corrupted_value"] = dirty_df.at[idx, "origin_port"]
             log_entry["severity"] = "medium"
             log_entry["detection_hint"] = "Cross-column dependency check"

        # Update expected review status since the data is now dirty
        dirty_df.at[idx, "expected_review_status"] = "Manual Review"
        
        error_log.append(log_entry)

    error_log_df = pd.DataFrame(error_log)
    return dirty_df, error_log_df

def main():
    parser = setup_argparse()
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    Faker.seed(args.seed)
    fake = Faker()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.rows} clean manifests...")
    clean_df = generate_clean_manifests(args.rows, fake)
    
    print(f"Injecting errors into ~{args.error_rate * 100}% of the data...")
    dirty_df, error_log_df = inject_errors(clean_df, args.error_rate, fake)

    # Save outputs
    clean_path = output_dir / "shipping_manifests_clean.csv"
    dirty_path = output_dir / "shipping_manifests_dirty.csv"
    error_path = output_dir / "shipping_manifest_error_log.csv"

    clean_df.to_csv(clean_path, index=False)
    dirty_df.to_csv(dirty_path, index=False)
    error_log_df.to_csv(error_path, index=False)

    print("\n--- Generation Summary ---")
    print(f"Total clean rows generated: {len(clean_df)}")
    print(f"Total dirty rows generated: {len(dirty_df)}")
    print(f"Total errors injected: {len(error_log_df)}")
    print("\nError counts by type:")
    print(error_log_df['error_type'].value_counts().to_string())
    print("\nOutput files saved to:")
    print(f"  - {clean_path}")
    print(f"  - {dirty_path}")
    print(f"  - {error_path}")

if __name__ == "__main__":
    main()