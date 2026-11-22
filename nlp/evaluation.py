"""
Comprehensive Model Evaluation & Unseen Sentence Verification Script.
"""

import json
import pandas as pd
from nlp.fatigue_classifier import DriverFatigueClassifier
from nlp.preprocessing import preprocess_text

def run_evaluation(metrics_path: str = "nlp/evaluation_results.json"):
    print("=" * 65)
    print("DROWZY NLP FATIGUE ANALYZER: EVALUATION REPORT & VERIFICATION")
    print("=" * 65)

    # 1. Print saved metrics
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        print(f"Total Dataset Samples: {metrics['dataset_total_samples']}")
        print(f"Training Samples:      {metrics['train_samples']}")
        print(f"Testing Samples:       {metrics['test_samples']}")
        print(f"Overall Accuracy:      {metrics['accuracy'] * 100:.2f}%")
        print(f"Macro Precision:       {metrics['precision_macro'] * 100:.2f}%")
        print(f"Macro Recall:          {metrics['recall_macro'] * 100:.2f}%")
        print(f"Macro F1-Score:        {metrics['f1_macro'] * 100:.2f}%\n")
        print("Per-Class Metrics:")
        print(f"{'Category':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
        print("-" * 60)
        cr = metrics['classification_report']
        for cat in metrics['classes']:
            cdata = cr[cat]
            print(f"{cat:<15} {cdata['precision']*100:>8.2f}%   {cdata['recall']*100:>8.2f}%   {cdata['f1-score']*100:>8.2f}%   {int(cdata['support']):>5}")
    except Exception as e:
        print(f"Could not load evaluation metrics file: {e}")

    # 2. Test unseen sentences across every category
    classifier = DriverFatigueClassifier()

    unseen_test_cases = [
        # Fatigue
        "I feel like my eyes are closing against my will.",
        "My eyelids are drooping and I am yawning every minute.",
        # Alert
        "I feel refreshed and ready for another hundred kilometers.",
        "Wide awake and paying full attention to the traffic.",
        # Rest Request
        "Let us take a brief stop at the upcoming service area.",
        "I want to take a break from driving right now.",
        # Stress
        "Traffic is unbearable and making me extremely irritable.",
        "Heavy rain and bad drivers are stressing me out completely.",
        # Neutral
        "The cruise control is holding the car at fifty miles per hour.",
        "Following route forty four towards the northern interchange."
    ]

    print("\n" + "=" * 65)
    print("TESTING UNSEEN GENERALIZATION SENTENCES (OUT-OF-SAMPLE):")
    print("=" * 65)
    for sent in unseen_test_cases:
        res = classifier.predict(sent)
        print(f"\nInput:        \"{sent}\"")
        print(f"Prediction:   {res['predicted_category']} (Confidence: {res['confidence_score']*100:.1f}%)")
        print(f"Tokens:       {res['preprocessing']['lemmatized_tokens']}")
        print(f"Advice:       {res['recommendation']}")

if __name__ == "__main__":
    run_evaluation()
