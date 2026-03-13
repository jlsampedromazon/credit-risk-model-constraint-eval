# PROJECT ABSTRACT

## 1. Context & Motivation

Credit risk models are central to consumer lending, determining whether applicants are approved for credit and under what terms. In regulated banking environments, these models must satisfy strict governance requirements related to interpretability, transparency, and economic coherence. To ensure that model outputs are explainable and consistent with financial intuition, institutions often impose structural constraints on predictive models. Two commonly used constraints are enforcing monotonic relationships between certain financial variables and default risk, and restricting models to additive structures that allow predictions to be decomposed into independent feature contributions. While these constraints improve regulatory acceptance and model interpretability, they may reduce the predictive flexibility of machine learning models. Understanding the trade-off between predictive performance and governance compliance is therefore an important challenge in applied machine learning for credit risk.

## 2. Research Question & Expected Contributions

This project investigates the following research question:

How much predictive performance is lost when credit risk models are constrained to satisfy additive and monotonic structural requirements commonly imposed in regulated banking environments?

To answer this question, the study compares constrained and unconstrained implementations of three model architectures widely used in credit risk modeling:

logistic regression
decision trees
gradient-boosted trees (XGBoost)

We will use the same dataset and feature set to train each model, comprised only of such features that are not restricted by regulation. By holding both the dataset and feature set constant across models, and introducing structural constraints only in the constrained versions, the project aims to isolate the predictive cost of the compliance constraints on credit modeling performance.

The expected contribution is an empirical comparison of predictive performance between compliance-constrained versus unconstrained implementations across the three model architectures, providing insight into the cost of compliance in credit assessment.

## 3. Regulatory Constraints to be Simulated

Two core structural constraints typically imposed by regulators on credit approval models within commercial banking will be simulated:

### Additive Structure

The model prediction must be expressible as the sum of independent feature effects:

This structure improves interpretability by allowing regulators to understand the marginal contribution of each variable to predicted default risk.

### Monotonic Relationships

Certain financial variables are expected to influence credit risk in economically consistent directions. For example:

* Higher income should not increase predicted default risk.
* Longer employment duration should not increase predicted risk.
* Greater credit bureau inquiry activity should not decrease predicted risk.

Monotonic constraints will be enforced on these relationships directly in the model.


While we are aware that compliance requirements around credit assessment in banking are far more extensive and intricate, we selected to implement additive structure and monotonicity constraints because, per our understanding, they represent two of the most widely used and significant governance constraints in regulated credit risk modeling. From a practical standpoint, we also considered the time and resource limitations of developing this project of our course, while still being able to meaningfully simulate the impact regulatory requirements on credit assessment. Our hypothesis is that they can be implemented directly in several widely used modeling frameworks, such as scorecard-style logistic regression and monotonic gradient boosting, allowing a controlled comparison between constrained and unconstrained models. We believe this scope makes it possible to isolate the predictive cost of governance-driven modeling restrictions while keeping the experimental design manageable and reproducible. Future work may involve exploring additional / more nuanced regulatory constraints.

## 4. Dataset and Features

The project uses the Home Credit Default Risk dataset, sourced from Kaggle, which contains anonymized consumer loan application data provided by the non-bank lender Home Credit. The analysis focuses on the application_train.csv dataset, which contains approximately 307,000 loan applications with a binary target variable indicating whether the borrower defaulted. We note that in Kaggle the dataset has already been split. We believe the train split in Kaggle is both sufficient and manageable for our purposes.

Only variables that are generally acceptable in regulated credit underwriting models will be included. The same feature set will be used for both constrained and unconstrained models in order to isolate the impact of structural modeling constraints. Future work may investigate training the unconstrained model versions with additional features to understand the impact of these being excluded for regulatory reasons (e.g. so social media activity). We have selected the following dataset features to be used across the models:

Selected Features

| Feature Name | Description |
|--------------|-------------|
| AMT_INCOME_TOTAL | Total annual income reported by the applicant. |
| AMT_CREDIT | Total amount of credit requested in the loan application. |
| AMT_ANNUITY | Periodic repayment amount required for the loan (loan installment). |
| AMT_GOODS_PRICE | Price of the goods being financed by the loan. |
| DAYS_BIRTH | Applicant age (measured in days prior to the application date. |
| DAYS_EMPLOYED | Number of days the applicant has been employed by their current employer). |
| DAYS_REGISTRATION | Number of days since the applicant registered at their current address. |
| DAYS_ID_PUBLISH | Number of days since the applicant’s identity document was issued or updated. |
| OWN_CAR_AGE | Age of the applicant’s car, which can proxy asset ownership. |
| CNT_CHILDREN | Number of children financially dependent on the applicant. |
| CNT_FAM_MEMBERS | Total number of family members in the applicant’s household. |
| FLAG_OWN_CAR | Indicator showing whether the applicant owns a car. |
| FLAG_OWN_REALTY | Indicator showing whether the applicant owns real estate. |
| NAME_INCOME_TYPE | Type of income source (e.g., working, pensioner, business). |
| NAME_EDUCATION_TYPE | Highest education level attained by the applicant. |
| NAME_FAMILY_STATUS | Marital or family status of the applicant. |
| NAME_HOUSING_TYPE | Applicant housing situation (owning, renting, etc.). |
| OCCUPATION_TYPE | Occupation category of the applicant. |
| ORGANIZATION_TYPE | Industry or type of organization where the applicant works. |
| EXT_SOURCE_1 | External credit risk score from a third-party provider. |
| EXT_SOURCE_2 | External credit risk score from another independent provider. |
| EXT_SOURCE_3 | External credit risk score from a third provider. |
| AMT_REQ_CREDIT_BUREAU_HOUR | Number of credit bureau inquiries in the last hour. |
| AMT_REQ_CREDIT_BUREAU_DAY | Number of credit bureau inquiries in the last day. |
| AMT_REQ_CREDIT_BUREAU_WEEK | Number of credit bureau inquiries in the last week. |
| AMT_REQ_CREDIT_BUREAU_MON | Number of credit bureau inquiries in the last month. |
| AMT_REQ_CREDIT_BUREAU_QRT | Number of credit bureau inquiries in the last quarter. |
| AMT_REQ_CREDIT_BUREAU_YEAR | Number of credit bureau inquiries in the last year. |

Monotonic constraints will be applied to selected variables where the direction of relationship with default risk is economically well understood, such as income, age, employment duration, and credit bureau inquiry counts.

## 5. Model Architectures (Constrained & Unconstrained)

Three model architectures commonly used in credit risk modeling will be evaluated:

### Logistic Regression

##### Unconstrained:
Standard logistic regression trained on the selected features.

##### Constrained:
Scorecard-style logistic regression with monotonic binning and additive structure.

### Decision Tree (CART)

##### Unconstrained:
Standard CART decision tree capable of learning nonlinear feature interactions.

##### Constrained:
Monotonic decision tree with restricted depth and splits that preserve monotonic relationships.

### Gradient Boosted Trees (XGBoost)

Unconstrained:
Standard XGBoost model allowing flexible nonlinear interactions between features.

##### Constrained:
We implement monotonicity constraints using XGBoost’s built-in monotone_constraints parameter, which forces the model predictions to move monotonically with respect to selected features. Otherwise, this version uses the same model and hyperparameters as the unconstrained version.

## 6. Evaluation & Metrics: Constrained vs. Unconstrained Performance
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

## 7. Tech Stack
This project leverages a combination of statistical modeling, machine learning frameworks, and data orchestration tools to compare constrained vs. unconstrained performance.

Core Modeling & Algorithms
  - Scikit-Learn: Used for Logistic Regression and Decision Tree implementations, including GridSearchCV for hyperparameter tuning.

  - XGBoost: Primary gradient boosting framework, utilizing the native monotone_constraints parameter for constrained modeling.

  - Statsmodels: For detailed statistical summaries and p-value analysis in Logistic Regression.

Data Processing & Analytics
  - Pandas & NumPy: Core libraries for data manipulation and feature engineering.

Visualization & Reporting
  - Matplotlib & Seaborn: For generating ROC/PR curves and feature importance plots.

  - SHAP (SHapley Additive exPlanations): To visualize how constraints shift feature contributions between models.

Environment & Deployment
  - Python 3.9+: Base programming language.

  - Jupyter Notebooks: For exploratory data analysis (EDA) and iterative model testing.

## 8. Work Plan

| Week | Tasks |
|-----|------|
| Week 3/16 | Pre-processing of dataset (cleaning and feature selection)<br>Train unconstrained versions of the models<br>Log Reg: Ankita<br>CART: Jianyu<br>XGBoost: Jose |
| Week 3/23 | Train constrained versions of the models |
| Week 3/30 | Run experiment and evaluate results |
| Week 4/6 | Write-up / presentation |
| 4/13 to 4/15 | Buffer |
