"""
DroneSense: Acoustic Drone Detection System
Module: src/explainability.py
Author: Swati (24PD39)
Description: Explainable AI (XAI) Engine using SHAP (Shapley Additive Explanations).
             Provides Global Beeswarm interpretations and Local Waterfall breakdown
             for acoustic feature attribution (MFCC harmonics, spectral roll-off, etc.).
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Any
import shap


class DroneExplainer:
    """
    Explainability framework for interpreting acoustic drone detection models.
    Supports Tree-based models (Random Forest, XGBoost) and Kernel-based models.
    """

    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        output_dir: str = "reports",
    ):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: Trained classifier (e.g., RandomForestClassifier).
            feature_names: List of acoustic feature names.
            output_dir: Directory where explanation figures are saved.
        """
        self.model = model
        self.feature_names = feature_names
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize TreeExplainer by default, with fallback to Explainer
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.explainer = shap.Explainer(self.model)

    def explain(self, X: np.ndarray) -> shap.Explanation:
        """
        Compute SHAP values for given input features.
        Handles multi-class / binary output dimensions cleanly.
        """
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = list(X.columns)
            X_arr = X.values
        else:
            X_arr = X

        shap_values = self.explainer(X_arr)

        # If binary classification returns 3D explanation (samples, features, 2 classes),
        # slice for Class 1 (Drone detection)
        if len(shap_values.shape) == 3 and shap_values.shape[-1] == 2:
            shap_values = shap_values[:, :, 1]

        # Attach feature names if provided
        if self.feature_names is not None:
            shap_values.feature_names = self.feature_names

        return shap_values

    def plot_global_beeswarm(
        self,
        shap_values: shap.Explanation,
        max_display: int = 15,
        save_plot: bool = True,
    ) -> plt.Figure:
        """
        Generate Global SHAP Beeswarm Plot showing acoustic feature impact across clips.
        """
        fig = plt.figure(figsize=(10, 7))
        shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
        plt.title("Global Acoustic Feature Impact (SHAP Beeswarm)", fontsize=13, pad=15)
        plt.tight_layout()

        if save_plot:
            path = os.path.join(self.output_dir, "shap_global_beeswarm.png")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            print(f"[+] Global Beeswarm plot saved: {path}")

        return fig

    def plot_local_waterfall(
        self,
        shap_values: shap.Explanation,
        sample_index: int = 0,
        max_display: int = 10,
        save_plot: bool = True,
    ) -> plt.Figure:
        """
        Generate Local SHAP Waterfall Plot explaining prediction for a single audio clip.
        """
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[sample_index], max_display=max_display, show=False)
        plt.title(f"Local Decision Breakdown: Sample #{sample_index}", fontsize=13, pad=15)
        plt.tight_layout()

        if save_plot:
            path = os.path.join(self.output_dir, f"shap_local_waterfall_sample_{sample_index}.png")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            print(f"[+] Local Waterfall plot saved: {path}")

        return fig

    def get_top_features_summary(
        self,
        shap_values: shap.Explanation,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Calculate mean absolute SHAP value per feature and return top contributors.
        """
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        feature_names = (
            self.feature_names
            if self.feature_names is not None
            else [f"Feature_{i}" for i in range(len(mean_abs_shap))]
        )

        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Mean_Abs_SHAP": mean_abs_shap}
        ).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

        summary_path = os.path.join(self.output_dir, "shap_feature_importance.csv")
        importance_df.to_csv(summary_path, index=False)
        print(f"[+] Top feature importance saved: {summary_path}")

        return importance_df.head(top_n)


# =====================================================================
# Standalone Execution / Simulation Mode
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("DroneSense: Running Explainable AI (SHAP) Analysis Engine")
    print("=" * 70)

    # Standard domain acoustic feature names extracted from audio clips
    ACOUSTIC_FEATURES = (
        [f"mfcc_{i}_mean" for i in range(1, 14)]
        + [f"mfcc_{i}_std" for i in range(1, 14)]
        + [
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff",
            "spectral_flatness",
            "zero_crossing_rate",
            "rms_energy",
        ]
    )

    processed_features_path = "data/processed/features_binary.csv"
    best_model_path = "best_drone_detector.joblib"

    if os.path.exists(processed_features_path) and os.path.exists(best_model_path):
        print("[+] Loading live acoustic dataset and best trained model...")
        df = pd.read_csv(processed_features_path)
        feature_cols = [c for c in df.columns if c not in ["label", "filename"]]
        X_sample = df[feature_cols].values[:200]
        feature_names = feature_cols
        model = joblib.load(best_model_path)
    else:
        print("[i] Live dataset/model not yet saved. Initializing simulated acoustic features...")
        from sklearn.ensemble import RandomForestClassifier

        np.random.seed(42)
        n_samples = 300
        n_features = len(ACOUSTIC_FEATURES)

        # Simulate acoustic feature distributions
        X_sim = np.random.randn(n_samples, n_features)
        # Give higher weight to motor harmonics (MFCC 1-3) & spectral roll-off to reflect real drone physics
        y_sim = (
            (X_sim[:, 0] * 1.5 + X_sim[:, 1] * 1.2 + X_sim[:, 28] * 1.8 + np.random.randn(n_samples)) > 1.0
        ).astype(int)

        feature_names = ACOUSTIC_FEATURES
        model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_sim, y_sim)
        X_sample = X_sim[:100]

    # Initialize Explainer
    drone_xai = DroneExplainer(model=model, feature_names=feature_names, output_dir="reports")

    print("[*] Computing SHAP values across acoustic features...")
    shap_vals = drone_xai.explain(X_sample)

    # 1. Global Beeswarm Interpretation
    print("[*] Generating Global SHAP Beeswarm Plot...")
    drone_xai.plot_global_beeswarm(shap_vals, max_display=15)

    # 2. Local Waterfall Interpretation (Sample 0)
    print("[*] Generating Local SHAP Waterfall Plot for single clip...")
    drone_xai.plot_local_waterfall(shap_vals, sample_index=0, max_display=10)

    # 3. Top Contributing Features Summary
    top_features = drone_xai.get_top_features_summary(shap_vals, top_n=10)
    print("\n" + "=" * 70)
    print("TOP ACOUSTIC DRONE INDICATORS (SHAP Importance):")
    print("=" * 70)
    print(top_features.to_string(index=False))

    print("\n[SUCCESS] SHAP explainability engine executed successfully!")