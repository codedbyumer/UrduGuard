"""
UrduGuard - Urdu-aware adversarial prompt detector

Uses character n-gram TF-IDF (script-agnostic: works for Urdu script AND
Roman Urdu without needing a pretrained multilingual transformer, which is
not feasible to download/fine-tune in this environment) + Logistic
Regression with class-weight balancing for the natural class imbalance
(malicious prompts are rare relative to benign traffic).

This keeps the same "class-imbalance-aware" design goal from the proposal
(focal-loss-style reweighting) while staying within a laptop-feasible
compute budget - a realistic constraint for an independent student project.
"""

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # urduguard/

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "urdu_adversarial_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "results", "urduguard_detector.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "results", "detector_metrics.json")


def main():
    df = pd.read_csv(DATASET_PATH)
    df["y"] = (df["label"] == "adversarial").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["y"], test_size=0.25, random_state=42, stratify=df["y"]
    )

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",     # character n-grams: robust to Urdu script + Roman Urdu spelling variation
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True,
    )
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    clf = LogisticRegression(
        class_weight="balanced",   # imbalance-aware, per proposal's design goal
        max_iter=2000,
        C=5.0,
    )
    clf.fit(Xtr, y_train)

    y_pred = clf.predict(Xte)
    y_proba = clf.predict_proba(Xte)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    report = classification_report(y_test, y_pred, target_names=["benign", "adversarial"])
    print("=== UrduGuard Detector - Test Set Performance ===")
    print(report)
    print("Confusion matrix [[TN, FP], [FN, TP]]:", cm)
    print(f"ROC-AUC: {auc:.3f}")

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump({"vectorizer": vectorizer, "clf": clf}, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
