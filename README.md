# 🎓 TDT4259 — Student Dropout Risk Prediction

This project predicts **student outcomes** — whether a student will *graduate*, *remain enrolled*, or *drop out* — using demographic and academic data.  
The goal is to detect at-risk students **early enough for intervention**, rather than purely maximizing accuracy.

> **Status:** The **runoff model** currently performs best overall, achieving **0.8949 accuracy** with XGBoost.  
> This notebook aligns most closely with the project’s goal of identifying students at risk of dropping out, even though it involves light dataset alterations.  
> **Dataset:** Included locally (`data/data.csv`) — originally from the [UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/dataset/697/).

---

## 🧠 Project Goal

- Predict each student's final academic outcome (*Graduated*, *Enrolled*, *Dropout*).  
- Explore multiple tree-based models: **XGBoost**, **LightGBM**, and **CatBoost**.  
- Address **class imbalance** using SMOTE and similar techniques.  
- Evaluate using **accuracy**, **macro F1-score**, and **confusion matrices**.  
- Examine **feature importance** to understand which factors help or hurt predictions.  
- Test the effect of removing low or negatively contributing features.

---

## 📓 Notebooks Overview

### `baseline.ipynb`
A minimal reference notebook.
- Establishes baseline metrics using simple classifiers.  
- Provides a performance benchmark for later comparisons.

---

### `main.ipynb`
The main modeling workflow.
- Applies SMOTE for class rebalancing.  
- Trains and compares **XGBoost**, **LightGBM**, and **CatBoost**.  
- Evaluates results with confusion matrices and feature importance plots.  
- Achieves up to **~0.78 accuracy** with CatBoost.

---

### `runoff.ipynb`
Focused on refinement and feature analysis.
- Uses **permutation importance** to identify harmful or noisy features.  
- Tests selective feature removal and adjusted dataset variants.  
- Achieves **0.8949 accuracy** using **XGBoost**, offering the most meaningful results for dropout-risk detection.  
- Currently considered the **best-performing notebook overall**.

---

## 🗂️ Repository Structure

```text
tdt4259/
├── data/
│   └── data.csv
├── baseline.ipynb
├── main.ipynb
├── runoff.ipynb
├── requirements.txt
└── README.md
```

---

## 📊 Results Summary

| Model Type | Notebook | Accuracy | Notes |
|-------------|-----------|-----------|--------|
| Baseline (simple) | `baseline.ipynb` | ~0.68 | Basic reference |
| XGBoost (main) | `main.ipynb` | ~0.77 | Strong overall |
| CatBoost (main) | `main.ipynb` | **~0.78** | Highest pure accuracy in main |
| XGBoost (runoff) | `runoff.ipynb` | **0.8949** | Best performing, aligns with dropout-risk detection goal |

*(Exact metrics and confusion matrices available in the notebook outputs.)*

---

## 📌 Notes

- **No additional preprocessing** (e.g., one-hot encoding or scaling) is required — features are already numerically encoded.  
- Random seeds ensure reproducibility, but minor variance is expected.  
- The **runoff model** achieves the best balance between accuracy and real-world interpretability for identifying students at risk of dropping out.

---

## 📄 License

This repository is part of the **TDT4259** course at NTNU.  
Dataset distributed under the original [UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/dataset/697/) license terms.
