# DataOrbit - Healthcare Provider Fraud Detection Project  
![CMS.gov](https://img.shields.io/badge/CMS.gov-005EA2?style=for-the-badge&logo=star-of-life&logoColor=white)

## 📌 Project Overview
Healthcare fraud costs the U.S. healthcare system over **$68 billion annually**. This project, commissioned by DataOrbit (simulated), aims to assist **Medicare** in detecting fraudulent healthcare providers using machine learning.  

Our goal is to develop a data-driven, interpretable pipeline that identifies high-risk providers while minimizing false positives. The system utilizes advanced classification models to analyze multi-table claims data, handling severe class imbalances to flag potential fraud effectively.

For a deep dive into the business context and technical requirements, please refer to the [Project Description](./docs/Project%20Description.pdf).

## 📊 Dataset Information

### Source
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)  

The dataset used in this project is the **Healthcare Provider Fraud Detection Analysis** dataset provided by *rohitrox* on Kaggle.  

**Link:** [Kaggle Dataset Source](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)

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

> **Note:** Detailed codebooks explaining every column (e.g., diagnosis codes, reimbursement definitions) can be found in `docs/DE 10 Codebook.pdf`.

## 💻 Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-77ACF1?style=for-the-badge&logo=seaborn&logoColor=white)
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
│   │   ├── CMS28_DESC_LONG_DX.txt                    # Long descriptions for diagnosis codes
│   │   ├── CMS28_DESC_LONG_SG.txt                    # Long descriptions for surgical procedures
│   │   ├── CMS28_DESC_SHORT_DX.txt                   # Short descriptions for diagnosis codes
│   │   └── CMS28_DESC_SHORT_SG.txt                   # Short descriptions for surgical procedures
│   ├── DE 10 Codebook.pdf                            # Detailed data dictionary & variable definitions
│   ├── FY_11_FR_County_to_CBSA_Xwalk.txt             # Crosswalk mapping counties to CBSA regions
│   ├── Project Description.pdf                       # Official project requirements & capstone details
│   └── code_definitions_grouped.csv                  # Grouped medical code definitions for engineering
├── models/                                           # Serialized machine learning models
│   ├── best_svm_fraud_model.pkl                      # Trained Support Vector Machine model
│   ├── fraud_detection_adaboost.pkl                  # Trained AdaBoost classifier
│   └── gradient_boosting.pkl                         # Trained Gradient Boosting model
├── notebooks/                                        # Jupyter notebooks for analysis & dev
│   ├── 01_data_exploration_and_feature_engineering.ipynb # EDA, cleaning & feature creation
│   ├── 02_modeling_Trial .ipynb                      # Decision Trees, Random Forest, Gradient Boosting
│   ├── 02_modeling_logistic+ada.ipynb                # Logistic Regression & AdaBoost
│   ├── 02_modeling_svm.ipynb                         # SVM model 
│   ├── 02_modeling_updated.ipynb                     # All models: DT, RF, GB, AdaBoost, LogReg, SVM
│   └── 03_evaluation2.ipynb                          # Final model evaluation & metric analysis
├── reports/                                          # Final project deliverables
│   ├── presentation.pptx                              # In-depth presentation/analysis slides
│   └── Technical Report.pdf                          # Comprehensive technical documentation
├── .gitignore                                        # Files to ignore in version control
└── README.md                                         # Project overview & instructions
````

## ⚙️ Methodology

1.  **Data Exploration & Engineering:**

      * Merged multi-source data (Inpatient + Outpatient + Beneficiary) by `Provider` and `BeneID`.
      * Created aggregated features: *Average Claim Amount*, *Count of Claims*, *Chronic Condition Scores*, and *Diagnosis Code Counts*.

2.  **Handling Imbalance:**

      * Addressed the 1:9 fraud ratio using resampling techniques and class-weighted learning to ensure the model captures minority class patterns.

3.  **Modeling:**

      * Tested multiple algorithms: **Support Vector Machines (SVM)**, **Logistic Regression**, **AdaBoost**, and **Gradient Boosting**.
      * The **Ensemble Approach (Logistic Regression + AdaBoost)** and **SVM** yielded the most robust results.

4.  **Evaluation:**

      * Optimized for **Recall** and **F1-Score** to minimize missed fraud cases (False Negatives).
      * Analysis includes Confusion Matrices and ROC-AUC curves available in `notebooks/03_evaluation2.ipynb`.

### 🏆 Model Comparison Table (Test Set Metrics)

| Model | Recall | Precision | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting** | 0.7746 | 0.4583 | 0.5759 | 0.9198 | 0.5233 |
| **Decision Tree** | 0.7465 | 0.4609 | 0.5699 | 0.8821 | 0.4582 |
| **Random Forest** | 0.6620 | 0.4608 | 0.5434 | 0.9187 | 0.5349 |
| **Logistic Regression** | 0.9155 | 0.3846 | 0.5417 | 0.9461 | 0.7113 |
| **AdaBoost** | 0.4366 | 0.7209 | 0.5439 | 0.8883 | 0.5537 |
| **SVM (Class Weight)** | 0.9079 | 0.3966 | 0.5520 | 0.9467 | 0.6886 |

*Metrics cited from Technical Report*

## 👥 Team Members

  * **Ali Tharwat**
  * **Amr Khaled**
  * **Mostafa Ahmed**
  * **Lakshy Rupani**

-----

*This project was conducted as part of the academic curriculum for the Machine Learning course (Winter 2025) at the German International University of Applied Sciences (GIU)*
