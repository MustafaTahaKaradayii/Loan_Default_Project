# Loan Default Prediction Project

### Author: Mustafa Taha Karadayi (r0879951)

## Project Overview:
This project focuses on predicting the likelihood of loan default using various machine learning models and evaluating their performance. The objective is to assist financial institutions in assessing the risk of loan default and improving decision-making processes.

### Dataset:
- **Source**: The dataset, named `Loan_Default.csv`, consists of 140,000 loan applications with 34 features before feature engineering. After feature engineering, the number of features is reduced to 14, and the number of rows is approximately 110,000 after outlier detection.
- **Target Variable**: The target variable is binary, represented by the column `Status`, where `0` means the loan is paid off, and `1` indicates a loan default.
  
### Goal:
- Achieve an **AUC score of at least 0.85** to reliably identify potential defaulters and mitigate financial risks.

---

## Project Structure:
The code is organized into several Python scripts and a dataset as follows:

- `main.py`: The entry point for the project. You may need to adjust the file path in this script.
- `feature_eng.py`: Contains feature engineering processes such as feature scaling, encoding, outlier removal, and dimensionality reduction.
- `evaluation.py`: Evaluates the models' performance using accuracy, F1 score, AUC, and other relevant metrics.
- `models.py`: Houses the machine learning models (e.g., Neural Networks, SVM, Random Forest, Gradient Boosting).
- `Loan_Default.xlsx`: The dataset used for the project.
- `Mustafa Taha Karadayi(report).docx`: A detailed report of the project and model performance.
- `Url to dataset.txt`: A text file with a URL link to the dataset.

---

## Machine Learning Models Used:

### 1. **Neural Networks (MLP)**
- Multiple configurations of the Multi-layer Perceptron (MLP) were tested with different hyperparameters, such as hidden layer sizes, activation functions, and solvers.
- Best Result: Accuracy = 0.92, F1 Score = 0.92, AUC = 0.98.

### 2. **Support Vector Machine (SVM)**
- Tuned using hyperparameters like `C`, `gamma`, and `kernel`.
- Best Result: Accuracy = 0.82, F1 Score = 0.77, AUC = 0.87.

### 3. **Gradient Boosting Classifier (GB)**
- Multiple configurations with different learning rates, number of estimators, and max depth.
- Best Result: Accuracy = 0.90, F1 Score = 0.90, AUC = 0.96.

### 4. **Random Forest Classifier (RF)**
- Tuned parameters such as `n_estimators`, `max_depth`, and `min_samples_split`.
- Best Result: Accuracy = 0.9084, F1 Score = 0.9066, AUC = 0.96.

### 5. **Enhanced Neural Network (with TensorFlow)**
- User-defined hyperparameters allowed for flexible tuning of the model.
- Best Result: Accuracy = 0.9017, F1 Score = 0.9017, AUC = 0.96.

---

## Data Preprocessing:
The preprocessing steps are crucial for ensuring high model performance. They include:
1. **Feature Engineering**:
   - Dropping irrelevant or constant columns.
   - Handling missing values with imputation.
   - Scaling numeric features using `StandardScaler` or `PowerTransformer`.
   - Encoding categorical variables using one-hot encoding.
   - Outlier detection and removal using the IQR method.
2. **Dimensionality Reduction**:
   - Performed via PCA and Random Forest-based feature importance to retain only the most relevant features.

---

## Results:
The model performances are summarized below:

| Model                        | Accuracy | F1 Score | AUC  | Average Precision (AP) |
|-------------------------------|----------|----------|------|------------------------|
| MLP (2nd Attempt)             | 0.97     | 0.97     | 1.00 | 0.99                   |
| MLP (3rd Attempt)             | 0.92     | 0.92     | 0.98 | 0.91                   |
| SVM (1st Attempt)             | 0.82     | 0.77     | 0.87 | 0.75                   |
| Gradient Boosting (1st Attempt)| 0.90     | 0.90     | 0.96 | 0.89                   |
| Random Forest (1st Attempt)    | 0.9084   | 0.9066   | 0.96 | 0.90                   |
| Enhanced NN (Set 3)           | 0.9017   | 0.9017   | 0.96 | 0.85                   |

### Best Model:
The **Enhanced Neural Network (Set 3)** was selected as the final model due to its high accuracy, minimal overfitting, and excellent generalizability.

---

## How to Run:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Loan_Default_Project.git
