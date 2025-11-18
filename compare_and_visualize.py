# compare_and_visualize.py
# ------------------------------------------------------------
# This file runs both Naive Bayes models, compares performance,
# and generates charts used by Streamlit.
# ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from naive_bayes_multinomial import run_multinomial_nb
from naive_bayes_gaussian import run_gaussian_nb

def run_full_evaluation():
    # Run both models
    acc_multi, report_multi, cm_multi = run_multinomial_nb()
    acc_gauss, report_gauss, cm_gauss = run_gaussian_nb()

    # Summary for Streamlit
    summary = {
        "Model": ["MultinomialNB", "GaussianNB"],
        "Accuracy": [acc_multi, acc_gauss],
        "Precision": [report_multi["1"]["precision"], report_gauss["1"]["precision"]],
        "Recall": [report_multi["1"]["recall"], report_gauss["1"]["recall"]],
        "F1-Score": [report_multi["1"]["f1-score"], report_gauss["1"]["f1-score"]],
    }

    summary_df = pd.DataFrame(summary)

    # -------- Comparison Chart --------
    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary_df, x="Model", y="Accuracy")
    plt.title("Accuracy Comparison")
    plt.savefig("comparison_chart.png", dpi=120)
    plt.close()

    # -------- Confusion Matrix Multinomial --------
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_multi, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - MultinomialNB")
    plt.savefig("confusion_matrix_multinomial.png", dpi=120)
    plt.close()

    # -------- Confusion Matrix Gaussian --------
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_gauss, annot=True, fmt="d", cmap="Greens")
    plt.title("Confusion Matrix - GaussianNB")
    plt.savefig("confusion_matrix_gaussian.png", dpi=120)
    plt.close()

    return {"summary_df": summary_df}
