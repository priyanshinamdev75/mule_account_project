# AI/ML-Based Classification of Suspicious Mule Accounts

## 📌 Project Overview

Banks are increasingly facing cyber-enabled financial fraud involving **mule accounts**. These accounts are used to receive, transfer, and move fraudulent funds through multiple transactions, making traditional rule-based fraud detection difficult.

This project aims to develop an **AI/ML-powered fraud detection system** that analyzes financial transaction data and identifies suspicious transaction patterns. Machine learning models are used to classify transactions as legitimate or fraudulent and generate a risk-based assessment.

---

## 🎯 Problem Statement

The objective of this project is to build a machine learning system capable of identifying suspicious financial transactions and potential mule-account activity.

The system analyzes transaction-level information such as:

- Transaction type
- Transaction amount
- Sender balance
- Receiver balance
- Sender and receiver account information
- Transaction frequency and behavioral patterns

The project focuses on:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Machine learning classification
- Model evaluation
- Risk prediction
- Fraud detection

---

## 📊 Dataset

The project uses a large-scale financial transaction dataset containing more than **6.3 million transactions** and **11 features**.

### Dataset Features

| Feature | Description |
|---|---|
| `step` | Time step associated with the transaction |
| `type` | Type of transaction such as PAYMENT, TRANSFER, CASH_OUT, etc. |
| `amount` | Amount involved in the transaction |
| `nameOrig` | Sender account identifier |
| `oldbalanceOrg` | Sender's balance before the transaction |
| `newbalanceOrig` | Sender's balance after the transaction |
| `nameDest` | Receiver account identifier |
| `oldbalanceDest` | Receiver's balance before the transaction |
| `newbalanceDest` | Receiver's balance after the transaction |
| `isFraud` | Target variable indicating whether the transaction is fraudulent |
| `isFlaggedFraud` | Existing system flag for potentially fraudulent transactions |

---

## 🔍 Data Preprocessing

The dataset is first inspected and cleaned before applying machine learning.

The preprocessing workflow includes:

1. Loading the dataset using Pandas
2. Checking dataset dimensions
3. Checking column names and data types
4. Checking missing values
5. Checking duplicate records
6. Checking categorical values
7. Detecting unusual or inconsistent values
8. Preparing data for feature engineering
9. Encoding categorical variables
10. Preparing features and target variables

### Initial Data Quality

- Rows: **6,362,620**
- Columns: **11**
- Missing values: **0**
- Duplicate rows: **0**

---

## 📈 Exploratory Data Analysis

EDA will be performed to understand patterns in fraudulent and legitimate transactions.

Important analysis includes:

- Fraud vs legitimate transaction distribution
- Transaction type distribution
- Transaction amount distribution
- Balance changes
- Fraud patterns by transaction type
- Sender and receiver behavior
- Relationship between transaction amount and fraud
- Class imbalance analysis

---

## ⚙️ Feature Engineering

Raw transaction data will be transformed into meaningful features that help machine learning models identify suspicious behavior.

Potential features include:

- Transaction amount
- Balance difference
- Sender balance change
- Receiver balance change
- Transaction type
- Transaction frequency
- Number of incoming transactions
- Number of outgoing transactions
- Total amount received
- Total amount transferred
- Average transaction amount
- Account activity patterns

These features help the model learn behavioral and transactional patterns associated with fraud.

---

## 🤖 Machine Learning Models

Multiple machine learning models will be trained and compared.

### Models

- Logistic Regression
- Random Forest
- XGBoost
- Isolation Forest for anomaly detection

The models will be evaluated using:

- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

Since fraud detection is generally an imbalanced classification problem, **Precision, Recall and F1-Score** will receive particular attention instead of relying only on accuracy.

---

## 🧠 ML Workflow

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Comparison
     ↓
Hyperparameter Tuning
     ↓
Best Model
     ↓
Save Model
