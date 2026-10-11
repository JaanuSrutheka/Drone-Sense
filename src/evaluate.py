"""
DroneSense: Acoustic Drone Detection System
Module: src/evaluate.py
Author: Swati (24PD39)
Description: Multi-Metric Model Evaluation Benchmark Suite.
             Evaluates classification algorithms across Accuracy, Balanced Accuracy,
             Precision, Recall, F1-Score, ROC-AUC, and PR-AUC (Average Precision).
             Generates confusion matrices, ROC/PR curves, and comparison leaderboards.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
)


class ModelEvaluator:
    """
    Evaluation suite for acoustic drone detection models.
    Supports single-model and multi-model benchmarking.
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the evaluator.
        
        Args:
            output_dir (str): Directory where plots and benchmark tables are saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate_single_model(
        self,
        model_name: str,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate a single classifier on the test dataset across 7 metrics.
        """
        # Predict class labels
        y_pred = model.predict(X_test)

        # Predict probability scores (or decision function values)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = y_pred.astype(float)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            roc_auc = float("nan")

        try:
            pr_auc = average_precision_score(y_test, y_proba)
        except ValueError:
            pr_auc = float("nan")

        metrics = {
            "Model": model_name,
            "Accuracy": round(acc, 4),
            "Balanced Accuracy": round(bal_acc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "PR-AUC": round(pr_auc, 4),
        }

        return metrics

    def benchmark_models(
        self,
        models_dict: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_csv: bool = True,
    ) -> pd.DataFrame:
        """
        Benchmark multiple models, construct a leaderboard, and save to CSV.
        """
        results = []
        for name, model in models_dict.items():
            print(f"[*] Evaluating {name}...")
            metrics = self.evaluate_single_model(name, model, X_test, y_test)
            results.append(metrics)

        leaderboard = pd.DataFrame(results)
        # Sort by F1-Score and ROC-AUC descending
        leaderboard = leaderboard.sort_values(
            by=["F1-Score", "ROC-AUC"], ascending=[False, False]
        ).reset_index(drop=True)

        if save_csv:
            csv_path = os.path.join(self.output_dir, "model_benchmark.csv")
            leaderboard.to_csv(csv_path, index=False)
            print(f"\n[+] Leaderboard saved to: {csv_path}")

        return leaderboard

    def plot_confusion_matrix(
        self,
        model_name: str,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_labels: Tuple[str, str] = ("Noise", "Drone"),
        save_plot: bool = True,
    ) -> plt.Figure:
        """
        Plot and save confusion matrix for a model using scikit-learn display.
        """
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(cmap="Blues", ax=ax, values_format="d")
        ax.set_title(f"Confusion Matrix: {model_name}")
        plt.tight_layout()

        if save_plot:
            clean_name = model_name.lower().replace(" ", "_")
            plot_path = os.path.join(self.output_dir, f"cm_{clean_name}.png")
            fig.savefig(plot_path, dpi=300)
            print(f"[+] Confusion matrix saved: {plot_path}")

        return fig

    def plot_roc_curves(
        self,
        models_dict: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_plot: bool = True,
    ) -> plt.Figure:
        """
        Plot comparative ROC curves for all benchmarked models on one chart.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        for name, model in models_dict.items():
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X_test)
            else:
                y_proba = model.predict(X_test).astype(float)

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
        ax.set_title("Comparative ROC Curves - Drone Acoustic Detection")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_plot:
            path = os.path.join(self.output_dir, "comparative_roc_curves.png")
            fig.savefig(path, dpi=300)
            print(f"[+] ROC curves saved: {path}")

        return fig

    def plot_pr_curves(
        self,
        models_dict: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_plot: bool = True,
    ) -> plt.Figure:
        """
        Plot comparative Precision-Recall curves (critical under 1:8 class imbalance).
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        for name, model in models_dict.items():
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X_test)
            else:
                y_proba = model.predict(X_test).astype(float)

            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
            ax.plot(recall, precision, lw=2, label=f"{name} (PR-AUC = {pr_auc:.3f})")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves (1:8 Class Imbalance)")
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_plot:
            path = os.path.join(self.output_dir, "comparative_pr_curves.png")
            fig.savefig(path, dpi=300)
            print(f"[+] PR curves saved: {path}")

        return fig


# =====================================================================
# Standalone Execution / Benchmark Simulation
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("DroneSense: Running Multi-Metric Model Evaluation Suite")
    print("=" * 70)

    # Initialize evaluator
    evaluator = ModelEvaluator(output_dir="reports")

    # Check if pre-extracted features and trained model exist
    processed_features_path = "data/processed/features_binary.csv"
    best_model_path = "best_drone_detector.joblib"

    if os.path.exists(processed_features_path) and os.path.exists(best_model_path):
        print("[+] Found production data and models. Evaluating live models...")
        df = pd.read_csv(processed_features_path)
        X = df.drop(columns=["label", "filename"], errors="ignore").values
        y = df["label"].values

        # Split evaluation sample
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = joblib.load(best_model_path)
        models_dict = {"Best Drone Detector": model}
    else:
        print("[i] Dataset / Trained models from audio pipeline not found yet.")
        print("[i] Initializing self-test simulation with benchmark models...")
        print("[i] Simulating 1:8 imbalance with 30 acoustic features...")

        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier

        # Generate synthetic acoustic dataset matching the 30 acoustic features
        np.random.seed(42)
        n_samples = 1000
        n_features = 30
        # 1:8 imbalance (approx 11% drone, 89% noise)
        X_test = np.random.randn(n_samples, n_features)
        y_test = (np.random.rand(n_samples) < 0.11).astype(int)

        # Baseline fits for benchmark demonstration
        models_dict = {
            "Logistic Regression": LogisticRegression().fit(X_test, y_test),
            "Support Vector Machine": SVC(probability=True).fit(X_test, y_test),
            "Decision Tree": DecisionTreeClassifier(max_depth=5).fit(X_test, y_test),
            "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42).fit(X_test, y_test),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5).fit(X_test, y_test),
        }

    # 1. Run Benchmark
    leaderboard = evaluator.benchmark_models(models_dict, X_test, y_test)
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE LEADERBOARD:")
    print("=" * 70)
    print(leaderboard.to_string(index=False))

    # 2. Generate Curves & Plots
    print("\n[*] Generating ROC & Precision-Recall curves...")
    evaluator.plot_roc_curves(models_dict, X_test, y_test)
    evaluator.plot_pr_curves(models_dict, X_test, y_test)

    # 3. Generate Confusion Matrix for Top Model
    top_model_name = leaderboard.iloc[0]["Model"]
    print(f"[*] Generating Confusion Matrix for Top Performer: {top_model_name}...")
    evaluator.plot_confusion_matrix(top_model_name, models_dict[top_model_name], X_test, y_test)

    print("\n[SUCCESS] Evaluation pipeline executed successfully!")