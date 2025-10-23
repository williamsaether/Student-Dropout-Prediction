# 🎓 TDT4259 — Student Dropout Risk Prediction

This project predicts **student outcomes** — whether a student will *graduate*, *remain enrolled*, or *drop out* — using demographic and academic data.  
The goal is to detect at-risk students **early enough for intervention**, rather than purely maximizing accuracy.

> **Status:** The **calibrated and threshold-tuned XGBoost** model currently performs best overall for identifying dropouts, reaching **0.889 accuracy**, **0.88 recall**, and **0.84 F1** on the *Dropout* class.  
> The **soft-vote ensemble** (XGBoost + CatBoost + LightGBM) achieves the **highest accuracy** at **0.8949**, while maintaining balanced precision and recall.  
> **Dataset:** Included locally (`data/data.csv`) — originally from the [UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/dataset/697/).

---

## 🧠 Project Goal

- Predict each student's final academic outcome (*Graduated*, *Enrolled*, *Dropout*).  
- Explore multiple gradient-boosted tree models: **XGBoost**, **LightGBM**, and **CatBoost**.  
- Address **class imbalance** via SMOTE, class weighting, and scale-pos-weight.  
- Evaluate models using **accuracy**, **F1**, **recall**, **ROC-AUC**, and **confusion matrices**.  
- Inspect **feature importances** and assess the effect of removing low-value or harmful features.  
- Calibrate probabilities and tune decision thresholds to better identify students at risk of dropping out.

---

## 📓 Notebooks Overview

### `baseline.ipynb`
A minimal reference notebook.
- Establishes baseline metrics using simple classifiers.  
- Serves as a benchmark for later models.  
- Accuracy around **0.68** across basic approaches.

---

### `main.ipynb`
Builds upon the baseline with full **feature engineering** and **gradient boosting models**.  
The dataset is multi-class (*Dropout*, *Enrolled*, *Graduate*), and each model is trained both with and without SMOTE to test class balance improvements.

**Highlights:**
- Adds ratio-based and average-grade engineered features to capture academic performance.  
- Tests multiple boosting frameworks:
  - **XGBoost:** ~0.7808 accuracy  
  - **LightGBM:** ~0.7763 accuracy  
  - **CatBoost:** **~0.7819 accuracy** (best overall in `main`)  
- Confirms that additional preprocessing (scaling, one-hot encoding) does not improve performance.  
- Demonstrates that gradient boosting handles the dataset’s encoding and structure effectively.

📈 *Conclusion:* CatBoost slightly outperformed the others in the full multi-class setup, reaching ~0.78 accuracy and strong balance across all classes.

---

### `runoff.ipynb`
Refines the task into a **binary dropout prediction** problem by merging “Enrolled” and “Graduate” into a single class — *Not Dropped Out*.  
This better aligns with the project’s primary goal: *detecting students at risk of dropping out.*

**Workflow improvements:**
- Adds **trend-based features** (`approval_diff`, `grade_diff`) capturing performance change.  
- Tests SMOTE rebalancing, feature cleaning, and **permutation importance** to identify harmful variables.  
- Tunes **XGBoost**, **CatBoost**, and **LightGBM** through randomized hyperparameter search.  
- Applies **isotonic calibration** and **threshold optimization** for best F1-recall trade-offs.  
- Introduces a small **feature-engineered variant (FEPlus)** including fail ratios.

**Results:**
- **XGBoost (tuned + calibrated + threshold):** 0.889 accuracy, 0.88 recall, 0.84 F1, ROC-AUC 0.887  
- **CatBoost (tuned + calibrated + threshold):** 0.878 accuracy, 0.89 recall, 0.82 F1  
- **Soft-vote Ensemble (XGB 0.6 / CAT 0.25 / LGB 0.15):** **0.8949 accuracy**, precision 0.86, recall 0.81, F1 0.83  
- Final models show significant improvement over multi-class results.

📈 *Conclusion:* The **XGBoost calibrated model** achieves the best recall and overall balance, while the **soft-vote ensemble** yields the highest accuracy.

---

### `runoff_catboost.ipynb`
A CatBoost-focused refinement of the binary dropout formulation.  
Explores class weighting, F1-optimized training, and probability calibration to maximize dropout detection performance.

**Workflow details:**
- Uses advanced feature engineering identical to `runoff`.  
- Performs **hyperparameter tuning** on depth, iterations, regularization, and bootstrap strategy.  
- Tests multiple optimization objectives:
  - **Standard tuning:** 0.8825 accuracy, 0.82 recall  
  - **F1-optimized CatBoost:** 0.8723 accuracy, 0.83 recall  
  - **Weighted dropout emphasis (1.2×):** 0.8847 accuracy, 0.82 recall  
- Final calibration + threshold selection improves recall to **0.89** and balanced F1 ≈ 0.82.

📈 *Conclusion:* CatBoost delivers consistent high recall, confirming its robustness for dropout-risk classification.  
When calibrated and tuned, it performs comparably to XGBoost, with slightly more conservative precision.

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

| Model Type         | Notebook         | Accuracy   | Recall (Dropout) | F1 (Dropout) | Notes                        |
|--------------------|------------------|------------|------------------|--------------|------------------------------|
| Baseline (simple)  | `baseline.ipynb` | ~0.76      | ~0.75            | ~0.77        | Reference only               |
| XGBoost (main)     | `main.ipynb`     | ~0.77      | ~0.76            | ~0.79        | Strong early result          |
| CatBoost (main)    | `main.ipynb`     | **~0.78**  | ~0.77            | ~0.79        | Highest in main              |
| XGBoost (runoff)   | `runoff.ipynb`   | **0.889**  | **0.88**         | **0.84**     | Calibrated + threshold tuned |
| CatBoost (runoff)  | `runoff.ipynb`   | 0.878      | **0.89**         | 0.82         | Slightly higher recall       |
| Soft-Vote Ensemble | `runoff.ipynb`   | **0.8949** | 0.81             | 0.83         | Highest overall accuracy     |

*(Exact metrics, ROC-AUC, and confusion matrices are shown in the notebook outputs.)*

---

## 📌 Notes

- Features are already numerically encoded; no additional scaling or one-hot encoding is required.  
- Random seeds are fixed for reproducibility.  
- The **runoff models**—especially **XGBoost (calibrated + threshold tuned)**—offer the best balance between accuracy, recall, and interpretability.  
- The **Soft-Vote Ensemble** provides slightly higher overall accuracy but slightly lower dropout recall.  
- **CatBoost (calibrated + threshold tuned)** remains a strong secondary model, achieving the highest recall (0.89) among all methods.

---

## 📄 License

This repository is part of the **TDT4259** course at **NTNU**.  
Dataset distributed under the original [UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/dataset/697/) license terms.
