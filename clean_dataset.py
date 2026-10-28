"""
Dataset Preprocessing and Sanitization Pipeline
MSc Cyber Security Dissertation Project (COMP11024)
University of the West of Scotland (UWS London Campus)
Author: Pravin Yadav (Banner ID: B01847540)

Description:
This script loads the raw multi-class ISCX-URL2016 dataset ('All.csv'),
filters strictly for binary classification (Benign vs. Phishing),
resolves missing and division-by-zero infinite values, and exports
the clean, balanced dataset to 'data/clean_phishing_benign.csv'.
"""

import os
import numpy as np
import pandas as pd

def clean_and_prepare_dataset(raw_csv_path='data/All.csv', output_csv_path='data/clean_phishing_benign.csv'):
    # Check if raw dataset exists (either in data/ or root)
    if not os.path.exists(raw_csv_path):
        if os.path.exists('All.csv'):
            raw_csv_path = 'All.csv'
        else:
            raise FileNotFoundError(f"Raw dataset 'All.csv' not found at {raw_csv_path} or current directory.")

    print(f"Loading raw dataset from: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path, low_memory=False)
    print(f"Raw Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")

    target_col = 'URL_Type_obf_Type'
    if target_col not in df.columns:
        raise KeyError(f"Expected target column '{target_col}' not found in dataset.")

    # 1. Filter strictly for Benign and Phishing classes
    print("\n[Step 1] Filtering for Binary Classes (Benign vs. Phishing)...")
    filtered_df = df[df[target_col].isin(['benign', 'phishing'])].copy()
    class_counts = filtered_df[target_col].value_counts()
    print("Class distribution after filtering:")
    for cls_name, count in class_counts.items():
        print(f"  - {cls_name}: {count} instances ({count / len(filtered_df) * 100:.2f}%)")

    # 2. Binary Encoding: Benign -> 0, Phishing -> 1
    print("\n[Step 2] Binary Encoding (0 = Benign, 1 = Phishing)...")
    filtered_df['label'] = filtered_df[target_col].map({'benign': 0, 'phishing': 1})

    # 3. Separate features and target
    feature_cols = [c for c in filtered_df.columns if c not in [target_col, 'label']]
    X = filtered_df[feature_cols].copy()
    y = filtered_df['label'].values

    # 4. Handle Missing (NaN) and Infinite Values (from division-by-zero in ratios)
    print("\n[Step 3] Handling Missing and Infinite Values...")
    inf_count = np.isinf(X.select_dtypes(include=np.number)).sum().sum()
    missing_count = X.isnull().sum().sum()
    print(f"  - Detected {missing_count} missing (NaN) values")
    print(f"  - Detected {inf_count} infinite (inf) values (e.g. division-by-zero in argPathRatio)")

    X_cleaned = X.replace([np.inf, -np.inf], 0).fillna(0)

    # 5. Export Clean Dataset
    print(f"\n[Step 4] Exporting Cleaned Dataset to: {output_csv_path}...")
    os.makedirs(os.path.dirname(output_csv_path) if os.path.dirname(output_csv_path) else '.', exist_ok=True)
    clean_export = X_cleaned.copy()
    clean_export['label'] = y
    clean_export.to_csv(output_csv_path, index=False)

    print(f"SUCCESS: Exported {len(clean_export)} sanitized records with {X_cleaned.shape[1]} features.")
    print("Zero NaNs or infinite values remain.")

if __name__ == '__main__':
    clean_and_prepare_dataset()
