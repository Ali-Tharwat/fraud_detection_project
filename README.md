# Fraud Detection Project

## Project Overview

This project applies machine learning to detect fraudulent healthcare providers using Medicare claims data. The main objective is to develop a robust predictive model capable of handling severe class imbalance and delivering transparent, explainable outputs suitable for real-world auditing and intervention.

Key elements of the project include:
- **Data exploration and feature engineering:** Several CSV files related to beneficiary, inpatient, outpatient, and labeled transactions are processed (`data/Train_Beneficiarydata.csv`, `data/Train_Inpatientdata.csv`, `data/Train_Outpatientdata.csv`, `data/Train_labels.csv`, etc.).
- **Modeling approaches:** Comparative modeling with algorithms such as Decision Trees,Random Forest,Logistic Regression,Gradient Boost, AdaBoost, SVM, and custom ensemble methods with detailed trial logs in the modeling notebooks.
- **Evaluation:** Models are validated with metrics such as accuracy, precision, recall, and F1-score, and results are documented in `notebooks/03_evaluation.ipynb`.
- **Documentation:** Domain-specific documentation and codebooks are provided in the `/docs` directory for deeper understanding and contextual reference.

## Team Members

- Ali Tharwat
- Amr Khaled
- Mostafa Ahmed
- Lakshy Rupani

## Summary of Results

- **Best Model Achieved:** The ensemble approach (Logistic Regression + AdaBoost) and SVM demonstrated superior fraud detection performance in trial runs.
- **Handling Imbalance:** Special attention was given to resampling techniques and custom evaluation metrics due to the rarity of fraudulent events in the data.
- **Results:** Models were evaluated using accuracy, precision, recall, and F1-score on a holdout test set. Ensemble models achieved notably higher F1-scores compared to baseline approaches, reflecting better detection of minority (fraudulent) cases.
- **Explainability:** Feature importance assessments and model interpretation are included in the notebooks.

For specific numerical outcomes and charts, refer to the summary cells in `notebooks/03_evaluation.ipynb`, and to the model checkpoints in the `notebooks/` directory.

## Reproduction Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ali-Tharwat/fraud_detection_project.git
   cd fraud_detection_project
   ```

2. **Set up your Python environment:**
   - Use Python 3.7 or above
   - Install package dependencies as specified by your notebooks (`pandas`, `scikit-learn`, `numpy`, `matplotlib`, etc.)
   - Install via pip:
     ```bash
     pip install pandas scikit-learn numpy matplotlib
     ```

3. **Prepare the data:**
   - Ensure that all raw and processed data files are present in the `data/` directory:
     - `Train_Beneficiarydata.csv`
     - `Train_Inpatientdata.csv`
     - `Train_Outpatientdata.csv`
     - `Train_labels.csv`
     - `final_train_data.csv`
     - `final_test_data.csv`
   - If needed, consult `docs/` for codebooks and file formats.

4. **Run the analysis:**
   - Start a Jupyter Notebook server:
     ```bash
     jupyter notebook
     ```
   - Open and execute `notebooks/01_data_exploration_and_feature_engineering.ipynb` to prepare data and features.
   - Proceed through the modeling notebooks (`02_modeling_logistic+ada.ipynb`, `02_modeling_svm.ipynb`, etc.).
   - Review results in `03_evaluation.ipynb`.

5. **Documentation:**
   - For detailed variable explanations and context, view files in `docs/` such as:
     - `Project Description.pdf`
     - `DE 10 Codebook.pdf`
     - `code_definitions_grouped.csv`
