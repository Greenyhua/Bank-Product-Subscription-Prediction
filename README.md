# Bank Product Subscription Prediction (LightGBM-centric)

Predict whether a customer will subscribe to a bank product using structured data and modern ML. We compare Decision Tree, Random Forest, XGBoost, and LightGBM, try stacking, and pick LightGBM for deployment based on performance, efficiency, and simplicity.

Full methodology and business insights are in Bank_Product_Subscription_Prediction_Report.pdf (In Chinese).

- Data: 30,000 records from Tianchi; target: subscribe (yes/no)
- Metrics: AUC (primary), F1 (secondary), plus accuracy and confusion matrix
- Final choice: LightGBM tuned by Optuna

## Highlights

- End-to-end pipeline:
    - LabelEncoder for target, OrdinalEncoder for categoricals, StandardScaler for numericals
    - 5-fold cross-validation
- Automated hyperparameter tuning (Optuna) for DT/RF/XGB/LGBM with F1 as objective
- Model comparison and stacking (RF + LGBM → LogisticRegression)
- Interpretability and visuals:
    - Classification report, confusion matrix
    - LightGBM feature importance (gain/split/cover)

Key drivers from the report: duration, emp_var_rate, pdays, campaign, month.

## Quick Start

1. Environment
- Python 3.9+
- Install: pandas, numpy, scikit-learn, lightgbm, xgboost, optuna, joblib, matplotlib, seaborn
2. Data
- Place train.csv and test.csv locally
- Index column: id
- Target: subscribe in {'yes','no'}
3. Run
- Steps in code/notebook:
    - Preprocess (encode + scale)
    - Optuna tuning per model (models saved as .pkl)
    - CV comparison (AUC)
    - Stacking experiments
    - Evaluation + plots (report, confusion matrix, feature importance)

Note: Keep variable names consistent when plotting LGBM importance (use the trained cla or the loaded lgbm_model) to avoid NameError.

## Results (from report and code)

- AUC (5-fold CV, mean):
    - RandomForest ≈ 0.8896
    - LightGBM ≈ 0.8871
    - XGBoost ≈ 0.8823
    - DecisionTree ≈ 0.7931
- Stacking (RF + LGBM + LR): AUC ≈ 0.8917 (small gain, higher complexity)
- Final model: LightGBM (best trade-off)
- Important features: duration, emp_var_rate, pdays, campaign, month
