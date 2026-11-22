"""
Driver Fatigue Text Classifier.

Provides high-level inference interface:
- Preprocesses raw input text
- Generates predicted category and confidence scores
- Provides tailored safety recommendations
- Preserves transparency with step-by-step preprocessing breakdown
"""

import os
import pickle
from typing import Dict, Any, Optional
from nlp.preprocessing import preprocess_text, demonstrate_preprocessing

# Safety recommendations mapping as defined in project specifications
SAFETY_RECOMMENDATIONS = {
    "FATIGUE": (
        "Fatigue-related indicators detected. Consider stopping at a safe location "
        "and resting before continuing your journey."
    ),
    "ALERT": (
        "No fatigue indicators were detected in the entered text. Continue to remain "
        "attentive and follow safe driving practices."
    ),
    "REST_REQUEST": (
        "You indicated that you need rest. Please stop at a safe location and take a break."
    ),
    "STRESS": (
        "Stress-related indicators detected. Consider taking a break at a safe location "
        "before continuing."
    ),
    "NEUTRAL": (
        "No clear fatigue-related indicators were identified in the text."
    )
}

CATEGORY_COLORS = {
    "FATIGUE": "#d90429",      # Bold Red
    "REST_REQUEST": "#f77f00", # Amber/Orange
    "STRESS": "#e63946",       # Crimson
    "ALERT": "#2a9d8f",        # Forest Green
    "NEUTRAL": "#457b9d"       # Steel Blue
}

SAFETY_DISCLAIMER = (
    "Notice: This prediction is an informational decision-support aid based on natural "
    "language processing and is not a clinical or medical diagnosis. Text predictions do not "
    "override physical webcam drowsiness warnings."
)


class DriverFatigueClassifier:
    """Inference wrapper for the NLP driver fatigue classification pipeline."""

    def __init__(self, model_path: str = "nlp/model.pkl"):
        self.model_path = model_path
        self.pipeline = None
        self._load_or_train()

    def _load_or_train(self):
        """Load trained model from disk, or train it if not present."""
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.pipeline = pickle.load(f)
        else:
            print(f"Model file not found at {self.model_path}. Training new model...")
            from nlp.train_model import train_and_evaluate
            _, self.pipeline = train_and_evaluate(model_save_path=self.model_path)

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classify input driver text.
        
        Returns:
            Dict containing:
            - input_text: original raw string
            - predicted_category: top predicted class
            - confidence_score: float (0.0 to 1.0)
            - class_probabilities: dict mapping each category to float probability
            - recommendation: contextual safety advice
            - category_color: hex color for UI rendering
            - preprocessing: dict showing each preprocessing step
            - disclaimer: safety notice
        """
        if not text or not str(text).strip():
            return {
                "input_text": text,
                "predicted_category": "NEUTRAL",
                "confidence_score": 0.0,
                "class_probabilities": {cat: 0.2 for cat in SAFETY_RECOMMENDATIONS.keys()},
                "recommendation": SAFETY_RECOMMENDATIONS["NEUTRAL"],
                "category_color": CATEGORY_COLORS["NEUTRAL"],
                "preprocessing": demonstrate_preprocessing(""),
                "disclaimer": SAFETY_DISCLAIMER
            }

        prep_breakdown = demonstrate_preprocessing(text)
        clean_text = prep_breakdown["final_processed_text"]

        # If preprocessing stripped all tokens, evaluate on raw lowered
        eval_text = clean_text if clean_text.strip() else text.lower()

        pred_class = self.pipeline.predict([eval_text])[0]
        probas = self.pipeline.predict_proba([eval_text])[0]

        class_labels = list(self.pipeline.classes_)
        prob_dict = {
            cls_name: float(probas[idx])
            for idx, cls_name in enumerate(class_labels)
        }

        confidence = prob_dict.get(pred_class, float(max(probas)))

        return {
            "input_text": text,
            "predicted_category": pred_class,
            "confidence_score": confidence,
            "class_probabilities": prob_dict,
            "recommendation": SAFETY_RECOMMENDATIONS.get(pred_class, SAFETY_RECOMMENDATIONS["NEUTRAL"]),
            "category_color": CATEGORY_COLORS.get(pred_class, "#457b9d"),
            "preprocessing": prep_breakdown,
            "disclaimer": SAFETY_DISCLAIMER
        }
