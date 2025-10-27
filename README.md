# 🎓 TDT4259 — Student Dropout Risk Prediction

This project predicts **student outcomes** — whether a student will *graduate*, *remain enrolled*, or *drop out* — using demographic and academic data.  
The goal is to detect at-risk students **early enough for intervention**, rather than purely maximizing accuracy.

> **Status:** The final **calibrated CatBoost runoff** model performs best overall for identifying dropouts, achieving **0.878 accuracy**, **0.89 recall**, and **0.82 F1** on the *Dropout* class.  
> The **tuned XGBoost runoff** model provides a close alternative with slightly higher precision and similar overall performance.  
> **Dataset:** Included locally (`data/data.csv`) — originally from the [UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/dataset/697/).

---

## 🧠 Project Goal

- Predict each student's final academic outcome (*Graduated*, *Enrolled*, *Dropout*).  
- Explore multiple gradient-boosted tree models: **XGBoost**, **LightGBM**, and **CatBoost**.  
- Address **class imbalance** using SMOTE and built-in class weighting.  
- Evaluate models using **accuracy**, **F1**, **recall**, **ROC-AUC**, and **confusion matrices**.  
- Inspect **feature importances** and **SHAP values** to interpret model behaviour.  
- Apply **probability calibration** and **threshold optimisation** for improved dropout detection.

---

## 📓 Notebooks Overview

### `baseline.ipynb`
A minimal reference notebook.
- Establishes baseline metrics using simple classifiers (Dummy, Logistic Regression, Random Forest).  
- Serves as a benchmark for later models.  
- Accuracy around **0.68–0.70** across basic approaches.

---

### `main.ipynb`
Extends the baseline with full **feature engineering** and **gradient boosting models**.  
The dataset is multi-class (*Dropout*, *Enrolled*, *Graduate*), and each model is trained both with and without SMOTE to assess class balance effects.

**Highlights:**
- Adds ratio-based and average-grade engineered features.  
- Tests multiple boosting frameworks:
  - **XGBoost:** ≈ 0.78 accuracy  
  - **LightGBM:** ≈ 0.77 accuracy  
  - **CatBoost:** **≈ 0.78 accuracy** (best in `main`)  
- Confirms that additional preprocessing (scaling, encoding) is unnecessary.  

📈 *Conclusion:* CatBoost slightly outperformed other frameworks in the multi-class setup with balanced precision and recall.

---

### `runoff.ipynb`
Reframes the problem into a **binary dropout classification** by merging “Enrolled” and “Graduate” into a single class — *Not Dropped Out*.  
This aligns directly with the goal of identifying students at risk.

**Workflow improvements:**
- Adds **trend-based features** (`approval_diff`, `grade_diff`) capturing semester-to-semester changes.  
- Uses SMOTE and class weighting for imbalance handling.  
- Applies **permutation importance** to identify harmful variables.  
- Tunes **XGBoost**, **LightGBM**, and **CatBoost** with randomized search.  
- Uses **isotonic calibration** and **threshold tuning** for better recall on the Dropout class.

**Results:**
- **XGBoost (tuned + calibrated + threshold):** 0.889 accuracy, 0.88 recall, F1 ≈ 0.84, ROC-AUC ≈ 0.87  
- **CatBoost (tuned + calibrated + threshold):** 0.878 accuracy, 0.89 recall, F1 ≈ 0.82, ROC-AUC ≈ 0.88  

📈 *Conclusion:* Both binary models outperform the multi-class versions. CatBoost provides the highest recall, while XGBoost yields marginally higher precision.

---

### `runoff_catboost.ipynb`
A CatBoost-focused refinement of the binary dropout task.  
Explores multiple optimisation strategies, class weighting, and calibrated decision thresholds.

**Workflow details:**
- Uses the same engineered features as `runoff`.  
- Tunes depth, iterations, regularisation, and bootstrap parameters.  
- Evaluates F1-optimised, weighted, and calibrated variants.  
- Final **calibrated CatBoost model** reaches **0.878 accuracy**, **0.89 recall**, and balanced **F1 ≈ 0.82**.  
- Achieves **ROC-AUC ≈ 0.88** on the test set.

📈 *Conclusion:* CatBoost delivers strong and stable recall, confirming its suitability for identifying at-risk students.  
Its SHAP-based explanations also make it the most interpretable option.

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

| Model Type       | Notebook              | Accuracy | Recall (Dropout) | Precision (Dropout) | F1 (Dropout) | ROC-AUC | Notes |
|------------------|-----------------------|-----------|------------------|---------------------|--------------|---------|-------|
| Baseline         | `baseline.ipynb`      | ~0.70     | ~0.72            | ~0.74               | ~0.73        | —       | Reference only |
| XGBoost (main)   | `main.ipynb`          | 0.78      | 0.76             | 0.81                | 0.79         | —       | Multiclass setup |
| CatBoost (main)  | `main.ipynb`          | **0.78**  | 0.77             | 0.79                | 0.78         | —       | Best multiclass |
| LightGBM (runoff) | `runoff.ipynb`       | 0.873     | 0.86             | 0.78                | 0.81         | 0.86    | Competitive baseline |
| XGBoost (runoff) | `runoff.ipynb`        | **0.889** | 0.88             | 0.80                | **0.84**     | 0.87    | Tuned + calibrated |
| CatBoost (runoff) (final) | `runoff_catboost.ipynb` | 0.878 | **0.89** | 0.77 | 0.82 | 0.88 | Final model |


*(All metrics are computed on the same stratified 80/20 test split.)*

---

## 📌 Notes

- Gradient boosting models clearly outperform traditional baselines for this task.  
- The **CatBoost (calibrated + threshold tuned)** model achieves the highest recall (0.89), making it the best choice for early dropout detection.  
- **XGBoost** performs similarly but emphasises slightly higher precision.  
- SHAP analysis shows that key predictors include **approval ratios**, **tuition fee status**, **course**, **grades**, and **age at enrolment**.  
- Figures for ROC curves, confusion matrices, and SHAP summaries are included in the notebooks.

---

## 📄 License

This repository is part of the **TDT4259 – Applied Data Science** course at **NTNU**.  
Dataset distributed under the original [UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/dataset/697/) license terms.
