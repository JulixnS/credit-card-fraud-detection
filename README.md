# Credit Card Fraud Detection System

## Project Overview
A machine learning application designed to detect fraudulent credit card transactions. 
This project addresses the challenge of **extreme class imbalance** (where fraud accounts for only 0.17% of data) by using **SMOTE** (Synthetic Minority Over-sampling Technique) and a tuned **Random Forest Classifier**.

The final product is deployed as an interactive **Streamlit Web App** that allows users to simulate transaction features and view real-time fraud probability.

## Key Features
* **Imbalance Handling:** utilized SMOTE to balance the training dataset, improving the model's ability to catch rare fraud cases.
* **Feature Optimization:** Reduced dimensionality from 30 to 10 key features (e.g., `V17`, `V14`, `V12`) using Random Forest Feature Importance to reduce noise.
* **Interactive Dashboard:** A Streamlit frontend for manual transaction testing and visualization.
* **Robust Metrics:** Optimized for **Recall (83%)** to minimize missed fraudulent transactions.

## Tech Stack
* **Python 3.x**
* **Machine Learning:** Scikit-Learn, Imbalanced-Learn (SMOTE)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit

## Model Performance
* **Recall:** 83% (Successfully identifies 83% of actual fraud cases)
* **Precision:** 72% (Low false positive rate)
* **Accuracy:** 99.9% (Note: Accuracy is misleading in imbalanced datasets; Recall is the priority metric).

## Project Structure
```text
credit-card-fraud-detection/
├── src/
│   ├── data_processing.py    # SMOTE and cleaning logic
│   ├── model.py              # Model training script
│   └── evaluation.py         # Confusion Matrix plotting
├── notebooks/                # Jupyter notebooks for initial EDA
├── app.py                    # Streamlit Frontend
├── requirements.txt          # Python dependencies
└── README.md                 # Project Documentation