# Healthcare Provider Fraud Detection  ![CMS.gov](https://img.shields.io/badge/CMS.gov-005EA2?style=for-the-badge&logo=star-of-life&logoColor=white) ![DataOrbit](https://img.shields.io/badge/Client-DataOrbit-0056D2?style=for-the-badge) 

## 📌 Project Overview
Healthcare fraud costs the U.S. healthcare system over **$68 billion annually**. This project, commissioned by DataOrbit (simulated), aims to assist **Medicare** in detecting fraudulent healthcare providers using machine learning.  

Our goal is to develop a data-driven, interpretable pipeline that identifies high-risk providers while minimizing false positives. The system utilizes advanced classification models to analyze multi-table claims data, handling severe class imbalances to flag potential fraud effectively.

## 📊 Dataset Information

### Source ![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)  **Link:** [Kaggle Dataset Source](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)

The dataset used in this project is the **Healthcare Provider Fraud Detection Analysis** dataset provided by *rohitrox* on Kaggle.  

### Dataset Specs
* **Provider Coverage:** Data includes over 5,400 providers.
* **Fraud Prevalence:** Highly imbalanced class distribution, with approximately **9-10%** of providers labeled as potentially fraudulent.

### File Descriptions
The raw data consists of four primary CSV files located in the `data/` directory:

| File Name | Description |
| :--- | :--- |
| **Train_Beneficiarydata.csv** | Patient demographics, insurance coverage, and chronic conditions (e.g., Alzheimer's, Diabetes). |
| **Train_Inpatientdata.csv** | Claims for patients admitted to hospitals, including admission dates, diagnosis codes, and reimbursement amounts. |
| **Train_Outpatientdata.csv** | Claims for hospital visits where patients were not admitted (tests, minor procedures). |
| **Train_labels.csv** | The target variable file linking `Provider` IDs to a binary fraud label (`Yes`/`No`). |

## 💻 Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?style=for-the-badge&logo=matplotlib)![Seaborn](https://img.shields.io/badge/Seaborn-%2377ACF1.svg?style=for-the-badge&logo=seaborn&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 📂 Project Structure

```text
fraud_detection_project/
├── data/                                             # Data storage
│   ├── Train_Beneficiarydata.csv                     # Patient demographics & chronic conditions
│   ├── Train_Inpatientdata.csv                       # Hospital admission claims data
│   ├── Train_Outpatientdata.csv                      # Hospital visit (non-admission) claims
│   ├── Train_labels.csv                              # Target labels (Fraud/Non-Fraud)
│   ├── final_test_data.csv                           # Processed test set for evaluation
│   └── final_train_data.csv                          # Processed training set for modeling
├── docs/                                             # Documentation & references
│   ├── CMSv28_Descriptions/                          # CMS diagnosis & procedure code references
│   ├── DE 10 Codebook.pdf                            # Detailed data dictionary & variable definitions
│   └── code_definitions_grouped.csv                  # Grouped medical code definitions for engineering
├── models/                                           # Serialized machine learning models
│   ├── best_svm_fraud_model.pkl                      # Trained Support Vector Machine model
│   ├── best_adaboost_fraud_model.pkl                 # Trained AdaBoost classifier
│   └── best_gradient_boosting_fraud_model.pkl        # Trained Gradient Boosting model
├── notebooks/                                        # Jupyter notebooks for analysis & dev
│   ├── 01_data_exploration...ipynb                   # EDA, cleaning & feature creation
│   ├── 02_modeling_updated.ipynb                     # Training pipeline: DT, RF, GB, AdaBoost, SVM
│   └── 03_evaluation2.ipynb                          # Final model evaluation & metric analysis
├── reports/                                          # Final project deliverables
│   ├── presentation.pptx                             # In-depth presentation/analysis slides
│   └── Technical Report.pdf                          # Comprehensive technical documentation
├── .gitignore                                        # Files to ignore in version control
└── README.md                                         # Project overview & instructions
````

## ⚙️ Methodology

1.  **Data Exploration & Feature Engineering:**

      * **Dimensionality Reduction:** Mapped thousands of raw ICD-9 diagnosis and procedure codes into **17 clinical "Super Groups"** (e.g., *Circulatory System*, *Renal Disease*, *Trauma*) to reduce sparsity and noise.
      * **Fraud-Specific Flags:** Engineered high-value features such as `IsPostDischargeBilling` (claims billed after discharge) and `ClaimDurationDays` (exact length of stay).
      * **Aggregation:** Consolidated data from beneficiary, inpatient, and outpatient files into a single **Provider-Level** dataset.

2.  **Handling Imbalance:**

      * Addressed the 9.7:1 class imbalance by conducting a comprehensive GridSearch across **8 strategies** (including SMOTE, ADASYN, and RandomUnderSampler).
      * **Result:** The pipeline utilizes robust class weighting and sampling techniques to ensure the model captures minority class patterns without overfitting.

3.  **Modeling:**

      * Evaluated multiple algorithms using 5-Fold Stratified Cross-Validation: **Support Vector Machines (SVM)**, **Gradient Boosting**, **AdaBoost**, **Random Forest**, and **Decision Trees**.
      * **Gradient Boosting** and **SVM** emerged as the top performers for different business objectives.

4.  **Evaluation:**

      * Optimized for **F1-Score** (balance) and **Recall** (minimizing missed fraud).
      * Performed a detailed financial analysis to quantify the business impact of each model.

### 🏆 Model Comparison Table (Test Set Metrics)

| Model | Recall | Precision | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting** | **0.8355** | 0.5142 | **0.6366** | 0.9458 | 0.6304 |
| **SVM (Class Weight)** | **0.9079** | 0.3966 | 0.5520 | **0.9467** | **0.6886** |
| **AdaBoost** | 0.6908 | **0.5585** | 0.6176 | 0.8505 | 0.4982 |

*(Source: Comprehensive Test Set Evaluation)*

### 💰 Business Impact & Cost Analysis

We performed a cost-sensitive evaluation assuming the following realistic constraints:

  * **Cost of Missed Fraud (FN):** $50,000 (Average fraud amount)
  * **Cost of Investigation (FP):** $1,000 (Admin cost)
  * **Savings per Caught Fraud (TP):** $500 (Admin savings) + Avoided Loss

| Model | Estimated Cost per Case | Net Savings | Fraud Losses Prevented |
| :--- | :--- | :--- | :--- |
| **SVM** | $603.20 | **$5,852,000\*\* | $6,900,000 |
| **Gradient Boosting** | $883.24 | $4,853,000 | $6,350,000 |
| **AdaBoost** | $1,531.42 | $2,712,000 | $5,250,000 |

> **Insight:** While **Gradient Boosting** offered the best F1-Score (balance), the **SVM** model proved to be the most *Cost-Effective* strategy, generating the highest Net Savings ($5.8M) by prioritizing Recall (90.8%) to catch the maximum number of fraudulent claims.

### 🔎 Key Fraud Indicators

Our **Gradient Boosting** model identified the following features as the strongest predictors of potential fraud:

1.  **ClaimDurationDays\_max:** The maximum length of a hospital stay associated with a provider. Fraudulent providers often bill for abnormally long stays.
2.  **AttPhys\_per\_Bene:** The ratio of attending physicians to beneficiaries. Unusual ratios can indicate network collusion.
3.  **DiagnosisGroupCode\_nunique:** The diversity of diagnosis codes used. Fraudulent providers may cycle through various codes to maximize reimbursement.

## 🚧 Limitations & Future Improvements

While the current system achieves high cost savings, we identified the following areas for growth:

  * **False Negatives:** The model currently misses \~16% of fraud cases, likely due to sophisticated fraud schemes that mimic legitimate billing patterns closely.
  * **Network Analysis:** Future iterations should incorporate graph theory to detect provider collusion rings (e.g., referral patterns between physicians), which our current tabular model does not capture.
  * **Temporal Features:** Integrating time-series analysis could help detect sudden spikes in billing activity or "burn-and-churn" schemes.

## 👥 Team Members

  * **Ali Tharwat**
  * **Amr Khaled**
  * **Mostafa Ahmed**
  * **Lakshy Rupani**

-----

*This project was conducted as part of the academic curriculum for the Machine Learning course (Winter 2025) at the German International University of Applied Sciences (GIU)*
