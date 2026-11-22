# Human Activity Recognition & Postural Transitions via Data Fusion

Comparative machine learning and statistical evaluation study classifying human physical activities and postural transitions from wearable triaxial inertial sensor streams in Python (`Scikit-Learn`, `imbalanced-learn`, `Pandas`, `NumPy`, `SciPy`).

📄 **Full Research Paper:** [Paper.pdf](./Paper.pdf) *(Spanish; see **Section 3.1** for my individual implementation)*  
📓 **Reproducible Notebook:** [Notebook.ipynb](./Notebook.ipynb) | [Static PDF Export](./Notebook%20-%20PDF%20version.pdf)

---

## Project Scope & Individual Contribution

As part of a semester-long comparative data-fusion study spanning three public HAR benchmarks, **I independently authored Section 3.1 (`Notebook.ipynb`)**, covering end-to-end EDA, time-window feature engineering, leak-free imbalanced classification pipelines, dimensionality reduction, classifier fusion, and non-parametric statistical testing on the **UCI Smartphone-Based Recognition of Human Activities and Postural Transitions** dataset.

---

## Dataset & Feature Engineering Pipeline

- **Source:** UCI Smartphone HAR & Postural Transitions (30 subjects, waist-mounted triaxial accelerometer + gyroscope sampled at **50 Hz**).
- **Raw Scale:** **815,614** sensor readings across 6 inertial axes and **12 target classes** (6 static/dynamic activities + 6 postural transitions exhibiting **~80% minority-class imbalance**).
- **Windowing:** Segmented continuous streams into non-overlapping **3-second windows** (`150` samples/window), yielding **5,437 feature windows** (`4,349` train / `1,088` test under an 80/20 split).
- **Engineered Features (`32` total — `16` per sensor):**
  - Per-axis mean, standard deviation, and maximum (`X, Y, Z`)
  - Pairwise inter-axis correlations (`X–Y`, `X–Z`, `Y–Z`)
  - Vector magnitude mean, standard deviation, Area Under Curve (AUC), and consecutive mean difference
- **Leak-Free Preprocessing (`imblearn.pipeline.Pipeline`):** `StandardScaler` and **Borderline-SMOTE** oversampling are encapsulated strictly inside each cross-validation fold so synthetic samples never leak into validation/test splits.

---

## Key Experimental Results (Section 3.1)

### 1. Base Classifiers — Default vs. Borderline-SMOTE & Tuned 10-Fold CV (`GridSearchCV`)

| Classifier | Test F1 (No SMOTE) | Test F1 (Borderline-SMOTE) | Tuned 10-Fold CV Score | Test Macro-F1 (12 Classes) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** (`n_estimators=350`) | `90.61%` | **`91.11%`** | **`90.94%`** | **`0.82`** |
| **Logistic Regression** (`L1, C=10, saga`) | `87.79%` | `87.32%` | `88.02%` | `0.73` |
| **Perceptron** (`L1, alpha=1e-3`) | `81.69%` | `81.80%` | `84.20%` | `0.66` |

### 2. Dimensionality Reduction & Feature Selection (`10-Fold CV Mean`)

| Classifier | PCA (95% Var.) | Kernel PCA | Model-Based Selection (`RFECV` / `SelectFromModel`) |
| :--- | :---: | :---: | :---: |
| **Random Forest** | `83.34%` | `83.39%` | **`85.73%`** (`SelectFromModel`) |
| **Logistic Regression** | `81.21%` | `81.21%` | **`85.16%`** (`RFECV`) |
| **Perceptron** | `68.66%` | `68.66%` | **`79.81%`** (`RFECV`) |

### 3. Classifier Fusion Strategies (`10-Fold Stratified CV` + Scaling + Borderline-SMOTE)

| Fusion Strategy | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Aggregation (Bagging)** | **`91.36%`** | **`91.75%`** | **`91.36%`** | **`91.44%`** |
| **Custom Voting** (LR + RF + Naive Bayes) | `89.30%` | `89.79%` | `89.30%` | `89.40%` |
| **AdaBoost** | `54.43%` | `65.26%` | `54.43%` | `48.21%` |

### 4. Statistical Significance Testing (Demšar Protocol)
- **Friedman Test:** `χ² = 19.5385`, **`p = 0.0001`** (`p < 0.05`) — rejects the null hypothesis of equal performance across classifiers.
- **Post-hoc Nemenyi Test:** Confirms Random Forest significantly outperforms Perceptron (**`p = 0.000039`**) and Logistic Regression modestly outperforms Perceptron (`p = 0.0495`), while Random Forest vs. Logistic Regression (`p = 0.1092`) exhibits a smaller statistical margin.

### Key Findings
- **Bagging beats boosting on noisy, oversampled sensor data:** Random Forest aggregation achieved the highest stability (`91.44% F1`), whereas AdaBoost degraded sharply (`48.21% F1`) by repeatedly up-weighting noisy/synthetic minority samples from Borderline-SMOTE.
- **Feature selection outperforms blind projection here:** Because the 32 engineered features were already compact and informative, `SelectFromModel` and `RFECV` preserved far more discriminative signal than PCA/Kernel PCA—while keeping all 32 features yielded the highest overall accuracy.

---

## Tech Stack & Running Locally

- **Core:** Python 3, `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `scipy`, `scikit-posthocs`, `matplotlib`, `seaborn`, `joblib`

```bash
pip install pandas numpy scikit-learn imbalanced-learn scipy scikit-posthocs matplotlib seaborn joblib jupyter
jupyter notebook Notebook.ipynb
```

---

## Repository Structure

```text
.
├── Notebook.ipynb             # Full Section 3.1 EDA, pipelines, CV, fusion & statistical tests
├── Notebook - PDF version.pdf # Rendered notebook with plots and confusion matrices
├── Paper.pdf                  # Full comparative team study (Section 3.1 authored by me)
└── README.md
```
