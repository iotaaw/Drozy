"""
Model Training and Evaluation Pipeline for Driver Fatigue Text Classification.

Uses:
- Custom Preprocessing (lowercasing, cleaning, tokenization, stopword removal, lemmatization)
- Feature Extraction: TF-IDF (unigrams + bigrams)
- Classifier: Multinomial Logistic Regression
- Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from nlp.preprocessing import preprocess_text

def train_and_evaluate(
    data_path: str = "nlp/dataset.csv",
    model_save_path: str = "nlp/model.pkl",
    metrics_save_path: str = "nlp/evaluation_results.json",
    cm_plot_path: str = "confusion_matrix.png",
    dist_plot_path: str = "dataset_distribution.png"
):
    print("=" * 60)
    print("DROWZY: Training Driver Fatigue Text Classification Model")
    print("=" * 60)

    # 1. Load dataset
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {len(df)} samples across {df['label'].nunique()} classes.")
    print("Class distribution:")
    print(df['label'].value_counts())

    # Save class distribution plot
    plt.figure(figsize=(8, 4.5))
    palette = ['#e63946', '#2a9d8f', '#e76f51', '#f4a261', '#457b9d']
    ax = sns.countplot(data=df, x='label', order=['FATIGUE', 'ALERT', 'REST_REQUEST', 'STRESS', 'NEUTRAL'], palette=palette)
    plt.title('Driver Fatigue Text Dataset Class Distribution', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Category', fontsize=11, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=11, fontweight='bold')
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f"Saved distribution plot to: {dist_plot_path}")

    # 2. Preprocess text
    print("\nPreprocessing text samples...")
    df['clean_text'] = df['text'].apply(preprocess_text)

    # 3. Train/Test split (80/20 stratified)
    X = df['clean_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set:  {len(X_test)} samples")

    # 4. Construct Scikit-Learn Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1500,
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
    ])

    # 5. Fit pipeline
    print("\nFitting TF-IDF Vectorizer and Logistic Regression Pipeline...")
    pipeline.fit(X_train, y_train)

    # 6. Evaluate on held-out test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    classes = sorted(list(pipeline.classes_))

    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
    prec_weighted = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1_mac = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
    f1_wt = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 50)
    print("TEST SET EVALUATION RESULTS:")
    print("=" * 50)
    print(f"Accuracy:          {acc * 100:.2f}%")
    print(f"Precision (Macro): {prec_macro * 100:.2f}%")
    print(f"Recall (Macro):    {rec_macro * 100:.2f}%")
    print(f"F1-Score (Macro):  {f1_mac * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(report_text)

    # 7. Compute and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar=True, annot_kws={"size": 12, "weight": "bold"})
    plt.title('Drowzy NLP Fatigue Classifier: Confusion Matrix', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted Class', fontsize=11, fontweight='bold')
    plt.ylabel('True Class', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_plot_path}")

    # 8. Save metrics to JSON
    metrics = {
        "dataset_total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "classes": classes,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict
    }

    with open(metrics_save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_save_path}")

    # 9. Save model pipeline
    with open(model_save_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Saved trained model pipeline to: {model_save_path}")

    print("\nTraining and evaluation completed successfully!")
    return metrics, pipeline

if __name__ == "__main__":
    train_and_evaluate()
