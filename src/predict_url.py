"""
Lightweight Phishing URL Inference Engine & Demonstration Script
MSc Cyber Security Dissertation Project (COMP11024)
University of the West of Scotland (UWS London Campus)
Author: Pravin Yadav (Banner ID: B01847540)
Supervisor: Dr. Raja Ujjan | Moderator: Dr. Muhsin Hassanu

This script demonstrates wire-speed, edge-viable phishing URL detection:
1. Ingests raw URL strings.
2. Extracts lexical and character-distribution features.
3. Maps to the trained Zero-Leakage Chi-Square top 20 feature space.
4. Generates instant binary verdict (BENIGN vs PHISHING), confidence score,
   and live inference latency.
"""

import os
import sys
import time
import re
import math
import joblib
import numpy as np
import pandas as pd

# Load serialized model bundle and top20 features
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'results')
MODEL_PATH = os.path.join(RESULTS_DIR, 'saved_models', 'decision_tree_redu.joblib')
RF_MODEL_PATH = os.path.join(RESULTS_DIR, 'saved_models', 'random_forest_redu.joblib')
TOP20_PATH = os.path.join(RESULTS_DIR, 'top20_chi2_features.csv')

def extract_lexical_features(url):
    """
    Extracts lexical characteristics from a raw URL string matching
    the ISCX-URL2016 lexical feature schema.
    """
    url_clean = str(url).strip()
    url_len = len(url_clean)
    
    # Split scheme, domain, path, query
    scheme_match = re.match(r'^(https?://)', url_clean)
    scheme = scheme_match.group(1) if scheme_match else ""
    without_scheme = url_clean[len(scheme):]
    
    parts = without_scheme.split('/', 1)
    domain = parts[0]
    path = '/' + parts[1] if len(parts) > 1 else ''
    
    # Domain tokens
    domain_tokens = re.split(r'[\.\-\_]', domain)
    domain_token_count = len(domain_tokens)
    
    # Delimiters
    delimiters = [';', '?', '=', '&', '#', '-', '_', '/', '.']
    delimeter_count = sum(url_clean.count(d) for d in delimiters)
    delimeter_path = sum(path.count(d) for d in delimiters)
    
    # Digits and symbols
    digits = sum(c.isdigit() for c in url_clean)
    NumberRate_URL = digits / max(url_len, 1)
    SymbolCount_Domain = sum(not c.isalnum() for c in domain)
    
    # Ratios
    domainUrlRatio = len(domain) / max(url_len, 1)
    
    # TLD length
    tld = len(domain_tokens[-1]) if len(domain_tokens) > 1 else 0
    
    # Feature dictionary
    feats = {
        'domainUrlRatio': domainUrlRatio,
        'domain_token_count': domain_token_count,
        'SymbolCount_Domain': SymbolCount_Domain,
        'tld': tld,
        'delimeter_path': delimeter_path,
        'delimeter_Domain': sum(domain.count(d) for d in delimiters),
        'NumberRate_URL': NumberRate_URL,
        'NumberRate_Domain': sum(c.isdigit() for c in domain) / max(len(domain), 1),
        'NumberRate_DirectoryName': sum(c.isdigit() for c in path) / max(len(path), 1),
        'NumberRate_AfterPath': 0.0,
        'Entropy_Domain': -sum((domain.count(c)/len(domain)) * math.log2(domain.count(c)/len(domain)) for c in set(domain)) if domain else 0.0,
        'Entropy_DirectoryName': -sum((path.count(c)/len(path)) * math.log2(path.count(c)/len(path)) for c in set(path)) if path else 0.0,
        'Entropy_URL': -sum((url_clean.count(c)/len(url_clean)) * math.log2(url_clean.count(c)/len(url_clean)) for c in set(url_clean)) if url_clean else 0.0,
        'Query_LetterCount': sum(c.isalpha() for c in path.split('?')[1]) if '?' in path else 0,
        'Query_digitCount': sum(c.isdigit() for c in path.split('?')[1]) if '?' in path else 0,
        'subDirMarker_URL': url_clean.count('/'),
        'pathurlRatio': len(path) / max(url_len, 1),
        'argPathRatio': 0.0,
        'domainlength': len(domain),
        'pathLength': len(path)
    }
    return feats

def classify_urls(test_urls):
    print("=" * 80)
    print("MSc CYBER SECURITY: REAL-TIME LIGHTWEIGHT PHISHING URL INFERENCE")
    print("Zero-Leakage Chi-Square (20 Features) | Candidate: Pravin Yadav (B01847540)")
    print("=" * 80)
    
    # Check model
    active_model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else RF_MODEL_PATH
    if not os.path.exists(active_model_path):
        print(f"[ERROR] Trained model file not found at {active_model_path}. Run run_final_experiments.py first.")
        return
        
    model = joblib.load(active_model_path)
    print(f"Loaded Production Model: {os.path.basename(active_model_path)}")
    print(f"Model Disk Footprint: {os.path.getsize(active_model_path) / 1024:.2f} KB")
    print("-" * 80)
    
    for url in test_urls:
        feats = extract_lexical_features(url)
        # Vectorize
        feat_vector = np.array(list(feats.values())).reshape(1, -1)
        
        # Benchmark live inference latency
        t0 = time.perf_counter()
        pred = model.predict(feat_vector)[0]
        if hasattr(model, 'predict_proba'):
            conf = model.predict_proba(feat_vector)[0][pred] * 100.0
        else:
            conf = 100.0
        latency_us = (time.perf_counter() - t0) * 1e6
        
        status = "MALICIOUS PHISHING" if pred == 1 else "LEGITIMATE BENIGN"
        action = "[BLOCK & QUARANTINE]" if pred == 1 else "[ALLOW TRAFFIC]"
        color = "\033[91m" if pred == 1 else "\033[92m"
        reset = "\033[0m"
        
        print(f"Target URL : {url}")
        print(f"Verdict    : {status} (Confidence: {conf:.1f}%)")
        print(f"Latency    : {latency_us:.2f} microseconds (μs)")
        print(f"SOC Action : {action}")
        print("-" * 80)

if __name__ == '__main__':
    sample_urls = [
        "https://www.google.com/search?q=machine+learning+security",
        "http://paypal-account-verification-login-portal.secure-update-cmd.net/cgi-bin/login.php?ssl=true",
        "https://www.uws.ac.uk/current-students/student-portal/",
        "http://192.168.1.105/bankofamerica/secure-login/update-credentials.html",
        "https://github.com/scikit-learn/scikit-learn"
    ]
    classify_urls(sample_urls)
