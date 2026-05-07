# Microstructural Changes in Subcortical White Matter Regions and their relationship with PASAT score in  Multiple Sclerosis: A Proof of Concept using Machine Learning

This repository contains the **machine learning regression pipeline** for predicting working memory performance in Multiple Sclerosis using Fractional Anisotropy biomarkers from subcortical white matter regions.

The objective is to evaluate Fractional Anisotropy (FA) features from subcortical U-fiber regions for **continuous prediction of working memory performance** using PASAT scores in Relapsing-Remitting Multiple Sclerosis patients.

---

## Pipeline Overview

![Pipeline Overview](Figure1.png)

---

## Repository Contents

- `ML_nestedCV.py` — Main machine learning pipeline with:
  - Sequential Feature Selection (SFS) with Random Forest
  - 15 regression models with hyperparameter tuning
  - 5-fold stratified cross-validation repeated 10 times
  - Model evaluation using MSE, MAE, and R-squared
  - SHAP explainability for model interpretation
  - Hierarchical clustering for feature correlation analysis
- `MS_FA_labels_PASAT.csv` — Processed dataset with FA features and PASAT scores
- `requirements.txt` — Python dependencies
- `Figure1.png` — Preprocessing pipeline visualization

---

## Methods Summary

**Modalities**: T1-weighted, Diffusion Tensor Imaging (DTI)  
**Subjects**: 93 subjects (35 Healthy Controls, 58 RRMS patients)  
**MRI Scanner**: Philips Ingenia 3T  
**Assessment**: Paced Auditory Serial Addition Test (PASAT) for working memory

Fractional Anisotropy features were extracted from 100 U-fiber regions using:
- DSI Studio for FA map computation
- SPM12 for co-registration and normalization
- LNAO-SWM79 Atlas for U-fiber region definition

### Feature Processing:
- Ridge regression for initial feature importance ranking
- Sequential Feature Selection (SFS) with Random Forest
- Optimal feature subset selection (16 features)
- Hierarchical clustering for feature correlation analysis

### Models Evaluated (15 regression algorithms):
- **Tree-based**: Random Forest, Gradient Boosting, XGBoost, CatBoost, Decision Tree, AdaBoost
- **Linear models**: Ridge, Lasso, Elastic Net, Bayesian Ridge, Logistic Regression, Quantile Regressor
- **Other**: SVM (RBF Kernel), K-Neighbors Regressor, MLP Regressor

### Model Selection & Validation:
- **GridSearchCV** with 5-fold stratified cross-validation
- **5-fold stratified cross-validation repeated 10 times** for robust performance estimation
- Hyperparameter optimization for each model
- Performance metrics: MSE, MAE, R-squared with mean ± standard deviation
- Best model: CatBoost 

### Interpretability:
- **SHAP analysis** for global and local feature importance
- Residual plots and regression diagnostics
- Hierarchical clustering visualization
- Feature importance rankings

---

## Reproducibility

- Python 3.10 (ML-pipeline)
- All random seeds fixed using `random`, `numpy`, `check_random_state`
- Classifiers use `random_state=42`
- StratifiedKFold CV is reproducible
- All model outputs and metrics exported to CSV

> MRI scans are not publicly available. Contact **Ethel Ciampi (ethelciampi@gmail.com)** for access or clinical inquiries.

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/pamelaFranco/continuous-prediction_ms_pasat.git
cd continuous-prediction_ms_pasat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the pipeline:
```bash
python ML_nestedCV.py
```

---

## License

MIT License

---

## Acknowledgements

This work is supported by the National Agency for Research and Development (ANID), project ICN2021_004 of the Millennium Science Initiative Program. Additional support provided by ANID through the project Fondecyt de Iniciación en Investigación 2025 Nº 11250867, and by the Competition for Research Regular Projects, year 2023, code LPR23-17, Universidad Tecnológica Metropolitana.