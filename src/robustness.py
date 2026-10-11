"""
DroneSense: Acoustic Drone Detection System
Module: src/robustness.py
Author: Swati (24PD39)
Description: Environmental Noise Robustness Stress-Testing Framework.
             Simulates real-world outdoor interference by injecting synthetic
             Additive White Gaussian Noise (AWGN) across SNR levels (+20dB down to -5dB).
             Evaluates model performance drop and plots Noise Degradation Curves.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class NoiseRobustnessTester:
    """
    Framework for stress-testing acoustic drone detection models
    under varying levels of environmental audio interference.
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the robustness tester.
        
        Args:
            output_dir (str): Directory where degradation plots and CSVs are saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def inject_awgn(X: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Inject Additive White Gaussian Noise (AWGN) into feature matrix based on target SNR (dB).
        
        Formula:
            SNR_dB = 10 * log10(P_signal / P_noise)
            P_noise = P_signal / (10 ^ (SNR_dB / 10))
            Noise ~ Normal(0, sqrt(P_noise))
        """
        X_arr = np.array(X, dtype=np.float64)

        # Calculate average signal power per sample
        signal_power = np.mean(X_arr ** 2, axis=1, keepdims=True)
        signal_power = np.maximum(signal_power, 1e-10)

        # Compute required noise power for target SNR
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))

        # Generate Gaussian noise
        noise = np.random.normal(0.0, np.sqrt(noise_power), size=X_arr.shape)

        return X_arr + noise

    def stress_test_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        snr_levels: Optional[List[float]] = None,
        save_csv: bool = True,
    ) -> pd.DataFrame:
        """
        Evaluate model stability across varying SNR levels from clean to severely degraded.
        """
        if snr_levels is None:
            # Standard project SNR sweep: +20dB to -5dB
            snr_levels = [20.0, 15.0, 10.0, 5.0, 0.0, -5.0]

        results = []

        # Baseline evaluation with 0 injected noise (Clean Audio)
        y_pred_clean = model.predict(X_test)
        results.append({
            "SNR_Level": "Clean (No Noise)",
            "SNR_dB": float("inf"),
            "Accuracy": round(accuracy_score(y_test, y_pred_clean), 4),
            "Balanced_Accuracy": round(balanced_accuracy_score(y_test, y_pred_clean), 4),
            "Precision": round(precision_score(y_test, y_pred_clean, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred_clean, zero_division=0), 4),
            "F1-Score": round(f1_score(y_test, y_pred_clean, zero_division=0), 4),
        })

        # Inject noise at each SNR level
        for snr in snr_levels:
            X_noisy = self.inject_awgn(X_test, snr_db=snr)
            y_pred = model.predict(X_noisy)

            results.append({
                "SNR_Level": f"{int(snr):+d} dB",
                "SNR_dB": float(snr),
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Balanced_Accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            })

        degradation_df = pd.DataFrame(results)

        if save_csv:
            csv_path = os.path.join(self.output_dir, "robustness_degradation_metrics.csv")
            degradation_df.to_csv(csv_path, index=False)
            print(f"[+] Degradation metrics saved: {csv_path}")

        return degradation_df

    def plot_degradation_curves(
        self,
        degradation_df: pd.DataFrame,
        model_name: str = "Drone Detector",
        save_plot: bool = True,
    ) -> plt.Figure:
        """
        Plot Accuracy, Balanced Accuracy, and F1-Score degradation curves across SNR levels.
        """
        curve_data = degradation_df[degradation_df["SNR_dB"] != float("inf")].copy()
        curve_data = curve_data.sort_values(by="SNR_dB", ascending=False)

        fig, ax = plt.subplots(figsize=(9, 6))

        snr_vals = curve_data["SNR_dB"].values
        f1_vals = curve_data["F1-Score"].values
        acc_vals = curve_data["Accuracy"].values
        bal_acc_vals = curve_data["Balanced_Accuracy"].values
        recall_vals = curve_data["Recall"].values

        # Plot metric trajectories
        ax.plot(snr_vals, f1_vals, marker="o", color="#d62728", lw=2.5, label="F1-Score")
        ax.plot(snr_vals, acc_vals, marker="s", color="#1f77b4", lw=2.0, linestyle="--", label="Accuracy")
        ax.plot(snr_vals, bal_acc_vals, marker="^", color="#2ca02c", lw=2.0, linestyle="-.", label="Balanced Accuracy")
        ax.plot(snr_vals, recall_vals, marker="d", color="#ff7f0e", lw=1.8, linestyle=":", label="Recall (Sensitivity)")

        ax.axhline(0.60, color="gray", linestyle="--", alpha=0.6, label="Critical Performance Threshold (60%)")

        ax.set_title(f"Environmental Noise Robustness: {model_name}\n(Performance under AWGN Interference)", fontsize=13, pad=12)
        ax.set_xlabel("Signal-to-Noise Ratio (SNR in dB) [High Quality -> Heavy Noise]", fontsize=11)
        ax.set_ylabel("Metric Score", fontsize=11)
        ax.set_xlim([max(snr_vals) + 1, min(snr_vals) - 1])  # Invert x-axis to show +20dB down to -5dB
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower left", framealpha=0.9)
        plt.tight_layout()

        if save_plot:
            plot_path = os.path.join(self.output_dir, "robustness_snr_degradation.png")
            fig.savefig(plot_path, dpi=300)
            print(f"[+] Degradation curve saved: {plot_path}")

        return fig


# =====================================================================
# Standalone Execution / Simulation Mode
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("DroneSense: Running Environmental Noise Robustness Framework")
    print("=" * 70)

    robustness_tester = NoiseRobustnessTester(output_dir="reports")

    processed_features_path = "data/processed/features_binary.csv"
    best_model_path = "best_drone_detector.joblib"

    if os.path.exists(processed_features_path) and os.path.exists(best_model_path):
        print("[+] Loading live test features and trained model artifact...")
        df = pd.read_csv(processed_features_path)
        feature_cols = [c for c in df.columns if c not in ["label", "filename"]]
        X = df[feature_cols].values
        y = df["label"].values

        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = joblib.load(best_model_path)
        model_name = "Trained Drone Detector"
    else:
        print("[i] Live dataset/model not yet saved. Initializing simulated test data...")
        from sklearn.ensemble import RandomForestClassifier

        np.random.seed(42)
        n_samples = 800
        n_features = 32

        # Simulate 1:8 class imbalance
        X_test = np.random.randn(n_samples, n_features)
        y_test = (np.random.rand(n_samples) < 0.11).astype(int)

        model = RandomForestClassifier(n_estimators=80, random_state=42).fit(X_test, y_test)
        model_name = "Random Forest Benchmark"

    # Run stress test across +20dB to -5dB
    print("[*] Simulating AWGN acoustic interference across SNR levels [+20dB to -5dB]...")
    degradation_df = robustness_tester.stress_test_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        snr_levels=[20.0, 15.0, 10.0, 5.0, 0.0, -5.0],
    )

    print("\n" + "=" * 70)
    print("ROBUSTNESS DEGRADATION TABLE:")
    print("=" * 70)
    print(degradation_df.to_string(index=False))

    # Plot and save curves
    print("\n[*] Generating Noise Degradation Curves...")
    robustness_tester.plot_degradation_curves(degradation_df, model_name=model_name)

    print("\n[SUCCESS] Environmental noise robustness framework executed successfully!")