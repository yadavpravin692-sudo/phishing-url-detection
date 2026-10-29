"""
Machine Learning Training, Feature Selection & Latency Profiling Pipeline
MSc Cyber Security Dissertation Project (COMP11024)
University of the West of Scotland (UWS London Campus)
Author: Pravin Yadav (Banner ID: B01847540)
Supervisor: Dr. Raja Ujjan | Moderator: Dr. Manesh Thankappan

Description:
This script executes the core empirical experiment:
1. Loads the sanitized binary dataset ('clean_phishing_benign.csv').
2. Enforces a zero-leakage 80:20 stratified train-test split (N_train=12,293, N_test=3,074).
3. Scales training features with MinMaxScaler to fulfill Chi-Square non-negative constraints.
4. Ranks features using Pearson's Chi-Square (χ²) statistic and extracts top k=20 features.
5. Trains and benchmarks 3 distinct classifiers (Random Forest, Decision Tree, Logistic Regression)
   across both Full (79 features) and Reduced (20 features) spaces.
6. Measures live inference latency (ms/URL), false negative rates, and training duration.
7. Exports metrics summary CSV and visualization figures.
"""

import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def run_pipeline():
    # 1. Locate cleaned dataset
    data_paths = ['clean_phishing_benign.csv', 'data/clean_phishing_benign.csv']
    dataset_path = None
    for p in data_paths:
        if os.path.exists(p):
            dataset_path = p
            break
            
    if not dataset_path:
        raise FileNotFoundError("Cleaned dataset 'clean_phishing_benign.csv' not found. Run clean_dataset.py first.")

    print(f"[STAGE 1] Loading Cleaned Dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Total instances: {df.shape[0]} | Total columns: {df.shape[1]}")

    feature_cols = [c for c in df.columns if c != 'label']
    X = df[feature_cols].copy()
    y = df['label'].values

    # 2. Zero-Leakage 80:20 Stratified Partitioning
    print("\n[STAGE 2] Stratified 80:20 Train-Test Split (Seed = 42)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train instances: {len(X_train)} (Phishing: {np.sum(y_train)}, Benign: {len(y_train)-np.sum(y_train)})")
    print(f"Test instances:  {len(X_test)} (Phishing: {np.sum(y_test)}, Benign: {len(y_test)-np.sum(y_test)})")

    # 3. Non-negative Scaling fitted strictly on training partition
    print("\n[STAGE 3] Fitting MinMaxScaler strictly on Training Partition...")
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    # 4. Chi-Square Feature Selection (k=20)
    print("\n[STAGE 4] Chi-Square (Chi^2) Feature Selection (Reducing 79 -> 20 features, -74.7%)...")
    k_selected = 20
    selector = SelectKBest(score_func=chi2, k=k_selected)
    X_train_reduced = selector.fit_transform(X_train_scaled, y_train)
    X_test_reduced = selector.transform(X_test_scaled)

    # Ranking
    chi2_scores = pd.DataFrame({
        'Feature': feature_cols,
        'Chi2_Score': selector.scores_
    }).sort_values(by='Chi2_Score', ascending=False)

    top_features = chi2_scores.head(k_selected)['Feature'].tolist()
    print("Top 10 Ranked Features by Chi-Square Score:")
    for rank, f_name in enumerate(top_features[:10], 1):
        score_val = chi2_scores.loc[chi2_scores['Feature'] == f_name, 'Chi2_Score'].values[0]
        print(f"  {rank:2d}. {f_name:<28} | Score: {score_val:8.2f}")

    # 5. Train & Benchmark Classifiers
    print("\n[STAGE 5] Training & Latency Benchmarking (Full 79 vs. Reduced 20 Features)...")
    models = {
        'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42), RandomForestClassifier(n_estimators=100, random_state=42)),
        'Decision Tree': (DecisionTreeClassifier(random_state=42), DecisionTreeClassifier(random_state=42)),
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), LogisticRegression(max_iter=1000, random_state=42))
    }

    results = []
    cm_dict = {}

    for name, (clf_full, clf_red) in models.items():
        print(f"\n--> Evaluating {name}...")
        
        # --- Full Features ---
        t0 = time.time()
        clf_full.fit(X_train_scaled, y_train)
        t_train_full = time.time() - t0

        t0 = time.time()
        y_pred_full = clf_full.predict(X_test_scaled)
        t_infer_full = (time.time() - t0) / len(X_test) * 1000  # ms/URL

        acc_full = accuracy_score(y_test, y_pred_full) * 100
        prec_full = precision_score(y_test, y_pred_full) * 100
        rec_full = recall_score(y_test, y_pred_full) * 100
        f1_full = f1_score(y_test, y_pred_full) * 100
        cm_full = confusion_matrix(y_test, y_pred_full)
        fn_full = cm_full[1, 0]

        # --- Reduced Features ---
        t0 = time.time()
        clf_red.fit(X_train_reduced, y_train)
        t_train_red = time.time() - t0

        t0 = time.time()
        y_pred_red = clf_red.predict(X_test_reduced)
        t_infer_red = (time.time() - t0) / len(X_test) * 1000  # ms/URL

        acc_red = accuracy_score(y_test, y_pred_red) * 100
        prec_red = precision_score(y_test, y_pred_red) * 100
        rec_red = recall_score(y_test, y_pred_red) * 100
        f1_red = f1_score(y_test, y_pred_red) * 100
        cm_red = confusion_matrix(y_test, y_pred_red)
        fn_red = cm_red[1, 0]

        cm_dict[f"{name} (Full)"] = cm_full
        cm_dict[f"{name} (Reduced)"] = cm_red

        results.append({
            'Classifier': name,
            'Features': 'Full (79)',
            'Accuracy': acc_full,
            'Precision': prec_full,
            'Phishing_Recall': rec_full,
            'F1_Score': f1_full,
            'False_Negatives': fn_full,
            'Latency_ms_URL': t_infer_full,
            'Train_Time_s': t_train_full
        })
        results.append({
            'Classifier': name,
            'Features': 'Reduced (20)',
            'Accuracy': acc_red,
            'Precision': prec_red,
            'Phishing_Recall': rec_red,
            'F1_Score': f1_red,
            'False_Negatives': fn_red,
            'Latency_ms_URL': t_infer_red,
            'Train_Time_s': t_train_red
        })

    results_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("EMPIRICAL BENCHMARK SUMMARY (MSc Cyber Security COMP11024)")
    print("="*80)
    print(results_df.to_string(index=False))

    # 6. Save Artifacts
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, 'preliminary_metrics_summary.csv')
    results_df.to_csv(summary_path, index=False)
    print(f"\n[Artifact] Saved metrics summary to: {summary_path}")

    # Generate Feature Importance Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=chi2_scores.head(15), x='Chi2_Score', y='Feature', palette='mako')
    plt.title('Top 15 Discriminative Features Ranked by Chi-Square (χ²)\n(Zero-Leakage Training Set: N=12,293)', fontsize=12, fontweight='bold')
    plt.xlabel('Chi-Square Test Statistic')
    plt.ylabel('Lexical URL Feature')
    plt.tight_layout()
    chart1 = os.path.join(results_dir, 'feature_importance_ranking.png')
    plt.savefig(chart1, dpi=300)
    plt.close()

    # Generate Confusion Matrices Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    model_order = [
        ('Random Forest (Full)', 'Random Forest (Reduced)'),
        ('Decision Tree (Full)', 'Decision Tree (Reduced)'),
        ('Logistic Regression (Full)', 'Logistic Regression (Reduced)')
    ]
    for col_idx, (m_full, m_red) in enumerate(model_order):
        sns.heatmap(cm_dict[m_full], annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0, col_idx])
        axes[0, col_idx].set_title(f"{m_full}\nFN: {cm_dict[m_full][1,0]}", fontweight='bold')
        axes[0, col_idx].set_xlabel('Predicted Label')
        axes[0, col_idx].set_ylabel('True Label')
        axes[0, col_idx].set_xticklabels(['Benign', 'Phishing'])
        axes[0, col_idx].set_yticklabels(['Benign', 'Phishing'])

        sns.heatmap(cm_dict[m_red], annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[1, col_idx])
        axes[1, col_idx].set_title(f"{m_red}\nFN: {cm_dict[m_red][1,0]}", fontweight='bold')
        axes[1, col_idx].set_xlabel('Predicted Label')
        axes[1, col_idx].set_ylabel('True Label')
        axes[1, col_idx].set_xticklabels(['Benign', 'Phishing'])
        axes[1, col_idx].set_yticklabels(['Benign', 'Phishing'])

    plt.suptitle('Confusion Matrix Comparison: Full 79 Features vs. Reduced 20 Features (Test Set: N=3,074)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    chart2 = os.path.join(results_dir, 'preliminary_confusion_matrices.png')
    plt.savefig(chart2, dpi=300)
    plt.close()

    # Generate Latency vs Accuracy Trade-off Chart
    plt.figure(figsize=(8, 5))
    for clf in ['Random Forest', 'Decision Tree', 'Logistic Regression']:
        sub = results_df[results_df['Classifier'] == clf]
        plt.plot(sub['Latency_ms_URL'], sub['Accuracy'], marker='o', linewidth=2, markersize=8, label=clf)
        for _, row in sub.iterrows():
            plt.annotate(row['Features'], (row['Latency_ms_URL'], row['Accuracy']), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9)

    plt.title('Inference Latency vs. Classification Accuracy Trade-off', fontsize=12, fontweight='bold')
    plt.xlabel('Runtime Inference Latency (Milliseconds per URL)')
    plt.ylabel('Classification Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    chart3 = os.path.join(results_dir, 'latency_vs_accuracy_tradeoff.png')
    plt.savefig(chart3, dpi=300)
    plt.close()

    print(f"[Artifact] Saved 3 visualization figures to '{results_dir}/'")
    print("Experiment completed successfully!")

if __name__ == '__main__':
    run_pipeline()
