"""
Complete Experimental Evaluation, Feature Selection, Profiling & Visualization Pipeline
MSc Cyber Security Final Dissertation Project (COMP11024)
University of the West of Scotland (UWS London Campus)
Author: Pravin Yadav (Banner ID: B01847540)
Supervisor: Dr. Raja Ujjan | Moderator: Dr. Muhsin Hassanu

This script:
1. Loads data/clean_phishing_benign.csv (15,367 samples, 79 lexical features).
2. Performs a strict Zero-Leakage 80:20 Stratified Train-Test split.
3. Fits MinMaxScaler and SelectKBest(chi2, k=20) ONLY on the training split.
4. Trains 5 classifiers across both Full (79) and Reduced (20) feature sets:
   - Random Forest (RF)
   - Decision Tree (DT)
   - Logistic Regression (LR)
   - Gaussian Naive Bayes (GNB)
   - Linear Support Vector Machine (Linear SVM)
5. Measures: Accuracy, Precision, Recall, F1, FPR, FNR, Specificity, ROC-AUC,
   Live Inference Latency (microseconds per URL), Training Time (seconds), Model Disk Footprint (KB).
6. Exports high-resolution publication figures (300 DPI) and comprehensive CSV summaries.
"""

import os
import sys
import time
import joblib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

def run_evaluation():
    print("=" * 80)
    print("MSc CYBER SECURITY: FINAL DISSERTATION EXPERIMENTAL PIPELINE")
    print("Candidate: Pravin Yadav (B01847540) | UWS London Campus")
    print("=" * 80)

    # 1. Output directories
    results_dir = os.path.abspath('03_final_dissertation/results')
    figures_dir = os.path.abspath('03_final_dissertation/figures')
    models_dir = os.path.join(results_dir, 'saved_models')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 2. Dataset loading
    csv_candidates = [
        'data/clean_phishing_benign.csv',
        'clean_phishing_benign.csv',
        '../data/clean_phishing_benign.csv'
    ]
    data_path = next((p for p in csv_candidates if os.path.exists(p)), None)
    if not data_path:
        raise FileNotFoundError("Could not find clean_phishing_benign.csv. Ensure data exists.")

    print(f"\n[STAGE 1] Loading Cleaned Dataset: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Total Rows: {len(df):,} | Total Columns: {df.shape[1]}")
    
    feature_cols = [c for c in df.columns if c != 'label']
    X = df[feature_cols].copy()
    y = df['label'].values
    print(f"Feature Space: {len(feature_cols)} lexical attributes")
    print(f"Class Distribution: Benign (0) = {(y == 0).sum():,} ({(y == 0).mean()*100:.2f}%), "
          f"Phishing (1) = {(y == 1).sum():,} ({(y == 1).mean()*100:.2f}%)")

    # 3. Strict Zero-Leakage 80:20 Stratified Partitioning
    print("\n[STAGE 2] Enforcing Zero-Leakage Stratified Split (80:20, Seed=42)")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training Partition: {len(X_train_raw):,} samples (Benign: {(y_train==0).sum():,}, Phishing: {(y_train==1).sum():,})")
    print(f"Testing Partition:  {len(X_test_raw):,} samples (Benign: {(y_test==0).sum():,}, Phishing: {(y_test==1).sum():,})")

    # 4. Normalization (MinMaxScaler) strictly fitted on Training Set
    print("\n[STAGE 3] Scaling Features (MinMaxScaler fitted strictly on Train Split)")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 5. Pearson's Chi-Square Feature Selection strictly fitted on Train Split
    K_FEATURES = 20
    print(f"\n[STAGE 4] Chi-Square (χ²) Feature Selection (Extracting Top k={K_FEATURES} Features)")
    selector = SelectKBest(score_func=chi2, k=K_FEATURES)
    X_train_reduced = selector.fit_transform(X_train_scaled, y_train)
    X_test_reduced = selector.transform(X_test_scaled)

    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_cols[i] for i in selected_indices]
    chi2_scores = selector.scores_[selected_indices]
    chi2_pvalues = selector.pvalues_[selected_indices]

    chi2_df = pd.DataFrame({
        'Feature': selected_features,
        'Chi2_Score': chi2_scores,
        'p_value': chi2_pvalues
    }).sort_values(by='Chi2_Score', ascending=False).reset_index(drop=True)

    chi2_export_path = os.path.join(results_dir, 'top20_chi2_features.csv')
    chi2_df.to_csv(chi2_export_path, index=False)
    print(f"Top 5 Chi-Square Features:\n{chi2_df.head(5).to_string(index=False)}")
    print(f"Exported full feature rankings to: {chi2_export_path}")

    # 6. Classifier Definitions
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'Decision Tree': DecisionTreeClassifier(max_depth=12, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Gaussian Naive Bayes': GaussianNB(),
        'Linear SVM': CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=2000, dual=False))
    }

    metrics_records = []
    confusion_dict = {}
    roc_dict = {}

    print("\n[STAGE 5] Benchmark Training & Live Profiling (Full vs Reduced)")
    print("-" * 80)

    for clf_name, clf_obj in classifiers.items():
        for space_name, X_tr, X_te, feat_count in [
            ('Full (79)', X_train_scaled, X_test_scaled, 79),
            ('Reduced (20)', X_train_reduced, X_test_reduced, 20)
        ]:
            model_key = f"{clf_name} [{space_name}]"
            print(f"Training: {model_key} ...", end=" ", flush=True)

            # Measure Training Time
            t_start = time.perf_counter()
            model = clone(clf_obj)
            model.fit(X_tr, y_train)
            t_train = time.perf_counter() - t_start

            # Measure Live Inference Latency over 10 repeats
            latencies = []
            for _ in range(10):
                t_inf_start = time.perf_counter()
                y_pred = model.predict(X_te)
                t_inf = time.perf_counter() - t_inf_start
                latencies.append((t_inf / len(X_te)) * 1e6) # microseconds per URL
            avg_latency_us = np.mean(latencies)

            # Predict Probabilities for ROC
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_te)[:, 1]
            else:
                y_proba = y_pred

            # Save Model and calculate disk size
            model_filename = f"{clf_name.lower().replace(' ', '_')}_{space_name[:4].lower()}.joblib"
            model_path = os.path.join(models_dir, model_filename)
            joblib.dump(model, model_path, compress=3)
            model_size_kb = os.path.getsize(model_path) / 1024.0

            # Metrics
            acc = accuracy_score(y_test, y_pred) * 100.0
            prec = precision_score(y_test, y_pred) * 100.0
            rec = recall_score(y_test, y_pred) * 100.0 # Sensitivity / Phishing Recall
            f1 = f1_score(y_test, y_pred) * 100.0
            
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            fpr = (fp / (fp + tn)) * 100.0
            fnr = (fn / (fn + tp)) * 100.0
            specificity = (tn / (tn + fp)) * 100.0

            fpr_arr, tpr_arr, _ = roc_curve(y_test, y_proba)
            roc_auc_val = auc(fpr_arr, tpr_arr) * 100.0

            confusion_dict[model_key] = cm
            roc_dict[model_key] = (fpr_arr, tpr_arr, roc_auc_val)

            metrics_records.append({
                'Classifier': clf_name,
                'Feature_Space': space_name,
                'Feature_Count': feat_count,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1_Score': f1,
                'FPR': fpr,
                'FNR': fnr,
                'Specificity': specificity,
                'ROC_AUC': roc_auc_val,
                'False_Positives': fp,
                'False_Negatives': fn,
                'Inference_Latency_us': avg_latency_us,
                'Train_Time_s': t_train,
                'Model_Size_KB': model_size_kb
            })
            print(f"Acc={acc:.2f}% | Rec={rec:.2f}% | Latency={avg_latency_us:.2f}μs | Size={model_size_kb:.1f}KB | Train={t_train:.2f}s")

    metrics_df = pd.DataFrame(metrics_records)
    summary_csv = os.path.join(results_dir, 'final_metrics_summary.csv')
    metrics_df.to_csv(summary_csv, index=False)
    print(f"\n[STAGE 6] Metrics successfully saved to: {summary_csv}")

    # 7. GENERATE HIGH-RESOLUTION FIGURES (300 DPI)
    print("\n[STAGE 7] Generating Publication-Grade Figures (300 DPI)...")
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'

    # Figure 1: 5x2 Confusion Matrices Grid
    fig, axes = plt.subplots(5, 2, figsize=(14, 22), dpi=300)
    fig.suptitle("Empirical Confusion Matrices: Full (79 Features) vs Reduced (20 Features)",
                 fontsize=16, fontweight='bold', y=0.995)
    
    clf_keys = list(classifiers.keys())
    for row_idx, clf_name in enumerate(clf_keys):
        for col_idx, space_name in enumerate(['Full (79)', 'Reduced (20)']):
            ax = axes[row_idx, col_idx]
            key = f"{clf_name} [{space_name}]"
            cm = confusion_dict[key]
            
            # Format annotation with count and % of total test
            total = cm.sum()
            annot = np.array([
                [f"TN\n{cm[0,0]:,}\n({cm[0,0]/total*100:.1f}%)", f"FP\n{cm[0,1]:,}\n({cm[0,1]/total*100:.1f}%)"],
                [f"FN\n{cm[1,0]:,}\n({cm[1,0]/total*100:.1f}%)", f"TP\n{cm[1,1]:,}\n({cm[1,1]/total*100:.1f}%)"]
            ])

            cmap = 'Blues' if col_idx == 0 else 'Greens'
            sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, cbar=False, ax=ax,
                        annot_kws={'fontsize': 11, 'fontweight': 'bold'})
            ax.set_title(f"{clf_name} — {space_name}", fontsize=12, fontweight='bold', pad=8)
            ax.set_xlabel("Predicted Class (0=Benign, 1=Phishing)", fontsize=10, labelpad=6)
            ax.set_ylabel("True Class", fontsize=10, labelpad=6)
            ax.set_xticklabels(['Benign', 'Phishing'], fontsize=10)
            ax.set_yticklabels(['Benign', 'Phishing'], fontsize=10, rotation=0)

    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, '01_confusion_matrices.png')
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Generated: {fig1_path}")

    # Figure 2: Comparative ROC Curves
    plt.figure(figsize=(11, 8), dpi=300)
    palette_full = {'Random Forest': '#1f77b4', 'Decision Tree': '#ff7f0e', 'Logistic Regression': '#2ca02c', 'Gaussian Naive Bayes': '#d62728', 'Linear SVM': '#9467bd'}
    for clf_name in clf_keys:
        # Full
        fpr_f, tpr_f, auc_f = roc_dict[f"{clf_name} [Full (79)]"]
        plt.plot(fpr_f, tpr_f, color=palette_full[clf_name], lw=2.0,
                 label=f"{clf_name} (Full 79) - AUC: {auc_f:.2f}%")
        # Reduced
        fpr_r, tpr_r, auc_r = roc_dict[f"{clf_name} [Reduced (20)]"]
        plt.plot(fpr_r, tpr_r, color=palette_full[clf_name], lw=1.8, linestyle='--',
                 label=f"{clf_name} (Reduced 20) - AUC: {auc_r:.2f}%")

    plt.plot([0, 1], [0, 1], color='#7f7f7f', lw=1.5, linestyle=':', label='Random Chance (AUC = 50.0%)')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold', labelpad=8)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, fontweight='bold', labelpad=8)
    plt.title('Receiver Operating Characteristic (ROC) Comparison\nSolid = Full Feature Set (79), Dashed = Reduced Feature Set (20)',
              fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc='lower right', fontsize=9.5, framealpha=0.95)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, '02_roc_curves.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Generated: {fig2_path}")

    # Figure 3: Top 20 Chi-Square Lexical Features Bar Chart
    plt.figure(figsize=(12, 8), dpi=300)
    chi2_sorted = chi2_df.sort_values(by='Chi2_Score', ascending=True)
    bars = plt.barh(chi2_sorted['Feature'], chi2_sorted['Chi2_Score'], color='#003366', edgecolor='#001f3f')
    plt.xlabel("Pearson's Chi-Square (χ²) Statistical Score (Fitted on Train Set)", fontsize=12, fontweight='bold', labelpad=8)
    plt.ylabel("Lexical Feature Attribute", fontsize=12, fontweight='bold', labelpad=8)
    plt.title("Top 20 Informative URL Lexical Features Selected by Chi-Square Filter\n(Zero-Leakage Selection Protocol)",
              fontsize=14, fontweight='bold', pad=12)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + (chi2_sorted['Chi2_Score'].max() * 0.01), bar.get_y() + bar.get_height()/2,
                 f"{w:,.1f}", va='center', ha='left', fontsize=9, fontweight='bold', color='#222222')
    plt.xlim(0, chi2_sorted['Chi2_Score'].max() * 1.14)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, '03_feature_importance_chi2.png')
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Generated: {fig3_path}")

    # Figure 4: Live Inference Latency vs Detection Accuracy (Pareto Frontier)
    plt.figure(figsize=(11, 7.5), dpi=300)
    for rec in metrics_records:
        marker = 'o' if 'Full' in rec['Feature_Space'] else 's'
        color = palette_full[rec['Classifier']]
        plt.scatter(rec['Inference_Latency_us'], rec['Accuracy'], s=150,
                    color=color, marker=marker, edgecolors='black', linewidth=1.2, zorder=5)
        
        offset_y = 0.3 if 'Full' in rec['Feature_Space'] else -0.4
        label_text = f"{rec['Classifier']}\n({rec['Feature_Space'][:7]}, {rec['Inference_Latency_us']:.1f}μs)"
        plt.annotate(label_text, (rec['Inference_Latency_us'], rec['Accuracy']),
                     textcoords="offset points", xytext=(8, offset_y*15),
                     fontsize=9, fontweight='bold', color='#111111',
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.85))

    plt.xscale('log')
    plt.xlabel('Live Inference Latency per URL (microseconds, log scale)', fontsize=12, fontweight='bold', labelpad=8)
    plt.ylabel('Overall Classification Accuracy (%)', fontsize=12, fontweight='bold', labelpad=8)
    plt.title('Lightweight Efficiency Frontier: Inference Latency vs Accuracy\nCircles = Full (79 Features), Squares = Reduced (20 Features)',
              fontsize=14, fontweight='bold', pad=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.ylim([80, 100])
    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, '04_latency_vs_accuracy_tradeoff.png')
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Generated: {fig4_path}")

    # Figure 5: Grouped Performance Metrics Across Classifiers (Full vs Reduced)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    metric_pairs = [
        ('Accuracy', 'Accuracy (%)', axes[0, 0]),
        ('Recall', 'Phishing Recall / Detection Rate (%)', axes[0, 1]),
        ('F1_Score', 'F1-Score (%)', axes[1, 0]),
        ('FPR', 'False Positive Rate (%) [Lower is Better]', axes[1, 1])
    ]
    for col, ylabel, ax in metric_pairs:
        piv = metrics_df.pivot(index='Classifier', columns='Feature_Space', values=col)
        # reorder index
        piv = piv.reindex(clf_keys)
        piv.plot(kind='bar', ax=ax, color=['#2b5c8f', '#48a970'], edgecolor='black', width=0.7)
        ax.set_title(ylabel, fontsize=12, fontweight='bold', pad=8)
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_xticklabels(clf_keys, rotation=25, ha='right', fontsize=9.5)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(['Full (79 Features)', 'Reduced (20 Features)'], fontsize=9)
        if col != 'FPR':
            ax.set_ylim([75, 101])
        else:
            ax.set_ylim([0, max(metrics_df['FPR'].max() * 1.25, 12)])
    plt.suptitle("Empirical Metric Comparison: Full 79 Lexical Features vs Reduced 20 Features",
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    fig5_path = os.path.join(figures_dir, '05_performance_grouped_bars.png')
    plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Generated: {fig5_path}")

    # Figure 6: Edge Resource Profiling (Model Size & Training Time)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Model size comparison
    size_piv = metrics_df.pivot(index='Classifier', columns='Feature_Space', values='Model_Size_KB').reindex(clf_keys)
    size_piv.plot(kind='bar', ax=ax1, color=['#b35806', '#f1a340'], edgecolor='black', width=0.7)
    ax1.set_title("Serialized Model Disk Footprint (KB, log scale)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Storage Size (KB)", fontsize=10, fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_xlabel('')
    ax1.set_xticklabels(clf_keys, rotation=25, ha='right', fontsize=9.5)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(['Full (79)', 'Reduced (20)'], fontsize=9)

    # Train time comparison
    time_piv = metrics_df.pivot(index='Classifier', columns='Feature_Space', values='Train_Time_s').reindex(clf_keys)
    time_piv.plot(kind='bar', ax=ax2, color=['#542788', '#998ec3'], edgecolor='black', width=0.7)
    ax2.set_title("Training Duration (seconds)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Training Time (s)", fontsize=10, fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_xticklabels(clf_keys, rotation=25, ha='right', fontsize=9.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(['Full (79)', 'Reduced (20)'], fontsize=9)

    plt.suptitle("Edge Feasibility Profiling: Model Footprint and Training Compute", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig6_path = os.path.join(figures_dir, '06_edge_resource_profiling.png')
    plt.savefig(fig6_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Generated: {fig6_path}")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED & VISUALIZATIONS EXPORTED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    run_evaluation()
