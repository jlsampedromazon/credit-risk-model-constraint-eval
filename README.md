# credit-risk-model-constraint-eval

## Evaluation & Metrics: Constrained vs. Unconstrained Performance
This section outlines the framework for comparing "Unconstrained" and "Constrained" versions of Logistic Regression, Decision Trees, and XGBoost

### Predictive Accuracy & Ranking
- These metrics evaluate how effectively the models differentiate between "Good" and "Bad" borrowers.

- ROC-AUC: Measures the overall ability to rank-order risk.

- PR-AUC: Critical for credit data where "Default" events are typically rare (imbalanced classes).

- K-S Statistic: The maximum separation between the cumulative distributions of defaults and non-defaults.

- Brier Score: Evaluates the accuracy of the raw Probability of Default (PD) estimates.

### Stability & Compliance
Assesses the reliability of the model logic under constraints.

- PSI (Population Stability Index): Monitors shifts in score distributions over time.

- Monotonicity Rate: Measures the percentage of features (e.g., Income) that maintain a logical, one-way relationship with risk.

- Feature Importance Alignment: Spearman’s Rank Correlation between the feature rankings of the two model versions to detect "logic shifts."

## Tech Stack
This project leverages a combination of statistical modeling, machine learning frameworks, and data orchestration tools to compare constrained vs. unconstrained performance.

1. Core Modeling & Algorithms
  - Scikit-Learn: Used for Logistic Regression and Decision Tree implementations, including GridSearchCV for hyperparameter tuning.
  
  - XGBoost: Primary gradient boosting framework, utilizing the native monotone_constraints parameter for constrained modeling.
  
  - Statsmodels: For detailed statistical summaries and p-value analysis in Logistic Regression.

2. Data Processing & Analytics
  - Pandas & NumPy: Core libraries for data manipulation and feature engineering.

3. Visualization & Reporting
  - Matplotlib & Seaborn: For generating ROC/PR curves and feature importance plots.

  - SHAP (SHapley Additive exPlanations): To visualize how constraints shift feature contributions between models.

4. Environment & Deployment
  - Python 3.9+: Base programming language.

  - Jupyter Notebooks: For exploratory data analysis (EDA) and iterative model testing.
