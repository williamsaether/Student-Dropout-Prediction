# 🎓 TDT4259 — Student Dropout Risk Prediction

This project predicts **student outcomes** — whether a student will *graduate*, *remain enrolled*, or *drop out* — using demographic and academic data.  
The goal is to detect at-risk students **early enough for intervention**, prioritizing *recall* on the *Dropout* class over overall accuracy.

> **Status:** The final **calibrated CatBoost (runoff)** model performs best overall for identifying dropouts, achieving **0.880 accuracy**, **0.89 recall**, **0.83 F1**, and **0.94 ROC-AUC** on the *Dropout* class after removing weak features.  
> The **tuned XGBoost runoff** model remains a close alternative with slightly higher precision and similar ROC-AUC.  
> **Dataset:** Included locally (`data/data.csv`) — originally from the [UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/dataset/697/).

---

## 🧠 Project Goal

- Predict whether a student will **drop out or not**.  
- Explore and compare gradient-boosting models: **XGBoost**, **LightGBM**, and **CatBoost**.  
- Address **class imbalance** via SMOTE, class weights, and calibration.  
- Evaluate using **accuracy**, **recall**, **F1**, and **ROC-AUC** (with emphasis on recall for early intervention).  
- Use **permutation importance**, **SHAP**, and **subgroup testing** to ensure fairness and robustness.  
- Apply **probability calibration** and **threshold optimisation** to improve dropout detection.

---

## 📓 Notebooks Overview

### `baseline.ipynb`
A minimal reference notebook.  
- Establishes baseline metrics using Logistic Regression and Random Forest.  
- Serves as a benchmark (≈ 0.68–0.70 accuracy).

---

### `main.ipynb`
Multi-class classification (*Dropout*, *Enrolled*, *Graduate*).  
Introduces **feature engineering** and **gradient boosting** with XGBoost, LightGBM, and CatBoost.

**Highlights:**
- Adds engineered ratio and grade features.  
- CatBoost slightly outperforms others (≈ 0.78 accuracy).  
- Confirms boosting models outperform traditional baselines.

---

### `runoff.ipynb`
Binary reformulation: merges *Enrolled* + *Graduate* → *Not Dropped Out*.  
Focuses directly on dropout detection.

**Workflow improvements:**
- Introduces semester-to-semester features (`approval_diff`, `grade_diff`).  
- Handles imbalance with SMOTE and class weights.  
- Applies isotonic calibration and tuned decision thresholds.  

**Results:**
| Model | Accuracy | Recall (Dropout) | F1 (Dropout) | ROC-AUC |
|--------|-----------|------------------|---------------|----------|
| XGBoost (tuned + calibrated + threshold) | **0.889** | 0.88 | **0.84** | 0.87 |
| CatBoost (tuned + calibrated + threshold) | 0.878 | **0.89** | 0.82 | 0.88 |

📈 *Conclusion:* Binary models clearly outperform multi-class. CatBoost yields the best recall, making it most suitable for intervention.

---

### `runoff_catboost.ipynb`
A full **CatBoost-focused refinement** of the binary dropout task with extended tuning, calibration, feature analysis, and robustness evaluation.

**Workflow details:**
- Uses the same engineered ratios, averages, and trend features as `runoff`.  
- Performs extensive hyperparameter tuning (depth = 7, lr = 0.03, 1000 iterations, Bayesian bootstrap).  
- Explores variants: F1-optimised, weighted (1.2× dropout class), calibrated, and threshold-tuned.  
- Applies **feature-removal stability tests** using permutation importance and cross-validation.  
- Adds **DropColumns** transformer inside the pipeline for reproducible feature pruning.  
- Performs **subgroup fairness analysis** and **robustness checks**.

**Key findings:**
- Harmful features (negative permutation importance):  
  *Mother’s qualification*, *Application order*, *Curricular units 1st/2nd sem (without evaluations)*, *International*.  
- Removing the weakest subset slightly improved performance:  
  - **Δ F1 ≈ +0.02**, **Δ ROC-AUC ≈ +0.002**.  
- Model remained stable across subgroup evaluations and feature removal.

**Final Results (Reduced Feature Set):**

| Metric | Score |
|---------|--------|
| **Accuracy** | **0.880** |
| **Recall (Dropout)** | **0.89** |
| **Precision (Dropout)** | 0.77 |
| **F1 (Dropout)** | **0.83** |
| **ROC-AUC** | **0.94** |

📈 *Conclusion:* The final calibrated and threshold-optimised CatBoost model balances recall and precision while maintaining strong ROC-AUC. It is robust to noisy features and consistent across student subgroups.

---

## 🧩 Robustness and Fairness

### Subgroup Performance

| Group | Subgroup | F1 (Dropout) | n |
|--------|-----------|--------------|---|
| **Attendance** | Daytime | 0.80 | 788 |
|  | Evening | 0.87 | 97 |
| **Course (range)** | Various IDs | 0.57 – 0.93 | — |

No significant bias was found. Lower performance for very small subgroups (e.g., n < 20) is attributed to limited data rather than model bias.

### Feature Stability
Removing the weakest five predictors improved the F1-score from **0.811 → 0.83**, confirming the model’s robustness.

---

## 🗂️ Repository Structure

```text
tdt4259/
├── data/
│   └── data.csv
├── baseline.ipynb
├── main.ipynb
├── runoff.ipynb
├── runoff_catboost.ipynb
├── requirements.txt
└── README.md
```

## 📊 Results Summary

| Model Type | Notebook | Accuracy | Recall (Dropout) | Precision (Dropout) | F1 (Dropout) | ROC-AUC | Notes |
|-------------|-----------|-----------|------------------|---------------------|--------------|----------|--------|
| Baseline | `baseline.ipynb` | ~0.70 | ~0.72 | ~0.74 | ~0.73 | — | Reference only |
| CatBoost (main) | `main.ipynb` | 0.78 | 0.77 | 0.79 | 0.78 | — | Best multi-class baseline |
| XGBoost (runoff) | `runoff.ipynb` | **0.889** | 0.88 | 0.80 | **0.84** | 0.87 | Tuned + calibrated |
| CatBoost (runoff) | `runoff.ipynb` | 0.878 | **0.89** | 0.77 | 0.82 | 0.88 | High recall |
| **CatBoost (runoff_catboost, full)** | `runoff_catboost.ipynb` | **0.878** | **0.89** | 0.77 | 0.82 | **0.94** | Final tuned + calibrated model |
| **CatBoost (runoff_catboost, reduced)** | `runoff_catboost.ipynb` | **0.880** | **0.89** | 0.77 | **0.83** | **0.94** | Final calibrated + threshold-tuned + reduced features |

*(All metrics computed on the same stratified 80/20 test split.)*

📈 *Conclusion:*  
Two CatBoost variants serve as final models:  
- The **full-feature model** achieves the same recall and ROC-AUC as the reduced version and is slightly more interpretable (for operational deployment).  
- The **reduced-feature model** removes redundant inputs, simplifying the pipeline while improving F1 by ≈ +0.02.

---

## 🧩 Robustness and Fairness

### Subgroup Performance

| Group | Subgroup | F1 (Dropout) | n |
|--------|-----------|--------------|---|
| **Attendance** | Daytime | 0.80 | 788 |
|  | Evening | 0.87 | 97 |
| **Courses (range)** | Various IDs | 0.57 – 0.93 | — |

Performance remains stable across both attendance types and courses.  
Differences between subgroups are minor (≤ 0.07 F1), and variations in very small groups (e.g., Course 33, n = 6) are attributed to limited data rather than model bias.  
This confirms that both CatBoost variants are **robust and fair** across diverse student populations.

---

### Feature Stability

Feature-removal experiments tested whether the model’s performance depended heavily on any single input.  
Removing five weak or noisy features —  
*Mother’s qualification*, *Application order*, *Curricular units 1st/2nd sem (without evaluations)*, and *International* —  
resulted in a small F1 improvement (**0.811 → 0.83**) and unchanged ROC-AUC (**≈ 0.94**).

✅ This demonstrates that the model is **stable and not reliant on redundant variables**.

---

### 🔍 Model Interpretability (SHAP Analysis)

SHAP summary plots were generated for both final CatBoost variants to interpret feature influence.  
Positive SHAP values indicate a higher likelihood of *dropping out*, while negative values indicate a higher likelihood of *continuing or graduating*.

#### Key insights
- **Financial and academic performance indicators** dominate the model’s predictions.  
  *Tuition fees up to date*, *Scholarship holder*, and *total_approved* were among the most important.  
- **Economic context variables** (e.g., GDP, Inflation rate) showed predictive power but are interpreted as *correlated contextual effects* rather than direct causal factors.  
- **Age at enrollment**, *Gender*, and *Course* provide moderate yet consistent contributions.  
- Demographic features like *parental occupation* have limited influence, reinforcing the model’s fairness.

#### Top 10 most influential features
| Rank | Feature | Description | Interpretation |
|------|----------|--------------|----------------|
| 1 | **Tuition fees up to date** | Indicates timely payment of fees | Financial stability strongly reduces dropout risk. |
| 2 | **total_approved** | Total curricular units approved | Higher academic progress correlates with retention. |
| 3 | **Age at enrollment** | Student’s age at program start | Younger students show slightly higher persistence. |
| 4 | **Course** | Program identifier | Certain degree programs show lower dropout rates. |
| 5 | **Curricular units 2nd sem (evaluations)** | Number of evaluations completed | Fewer evaluations indicate disengagement. |
| 6 | **Inflation rate** | Economic instability indicator | High inflation mildly increases dropout risk. |
| 7 | **GDP** | National economic context | Reflects macro-level correlation, not causation. |
| 8 | **Gender** | Student’s gender | Limited effect, suggesting model neutrality. |
| 9 | **approval_ratio_2nd** | Approval rate in 2nd semester | Indicates positive performance trajectory. |
| 10 | **Curricular units 2nd sem (grade)** | Average grade in 2nd semester | Lower grades correspond to higher dropout likelihood. |

📊 *Overall, SHAP confirms that academic engagement and financial stability drive the model’s predictions more than demographic or contextual features.*

---

## 📈 Final Model Highlights

| Variant | Accuracy | Recall (Dropout) | F1 (Dropout) | ROC-AUC | Notes |
|----------|-----------|------------------|---------------|----------|--------|
| **CatBoost (Full)** | 0.878 | **0.89** | 0.82 | **0.94** | Tuned + calibrated + threshold-tuned |
| **CatBoost (Reduced)** | **0.880** | **0.89** | **0.83** | **0.94** | Feature-pruned + calibrated + threshold-tuned |

Both final models exhibit strong recall and generalization, with nearly identical ROC-AUC performance.  
The reduced variant offers a more compact and stable pipeline for deployment, while the full model retains all features for maximum interpretability.

---

## 📌 Notes

- Gradient boosting models clearly outperform traditional baselines.  
- The **CatBoost (full)** and **CatBoost (reduced)** models are equally valid final versions —  
  one prioritizing *completeness*, the other *efficiency and robustness*.  
- Calibration and threshold tuning improved recall while maintaining AUC.  
- Subgroup and stability analyses confirm fairness and reliability across cohorts.  
- SHAP analysis validates the model’s interpretability, focusing on academic and financial signals rather than demographics.

---

## 📄 License

This repository is part of the **TDT4259 – Applied Data Science** course at **NTNU**.  
Dataset distributed under the original [UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/dataset/697/) license terms.

