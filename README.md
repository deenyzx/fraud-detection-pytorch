# fraud-detection-pytorch

> **Status: Pre-development** — README written as part of project planning. Implementation not yet started.

# Credit Card Fraud Detection with Autoencoder

A PyTorch-based anomaly detection system that identifies fraudulent credit card transactions using an autoencoder neural network.

---

## Overview

This project frames fraud detection as an **unsupervised anomaly detection** problem. The autoencoder is trained exclusively on legitimate transactions and learns to reconstruct normal behavior. Fraudulent transactions, being structurally different, produce a high reconstruction error — which is used as the anomaly score.

---

## Dataset

[Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions
- 492 fraud cases (~0.17% of all transactions)
- Features include: transaction amount, account balance, merchant info, card present flag, POS entry mode, and more

> Download the dataset from Kaggle and place it in the `data/` directory before running.

---

## How It Works

1. **Preprocessing** — Numerical features are scaled, categorical features are one-hot encoded, and new features are engineered (e.g. CVV match, account age, transaction hour)
2. **Training** — The autoencoder is trained **only on non-fraudulent transactions**, learning what normal looks like
3. **Inference** — Reconstruction error (MSE) is computed for every transaction
4. **Thresholding** — Transactions exceeding a chosen error threshold are flagged as fraud
5. **Evaluation** — Model is assessed using AUROC, AUPRC, and F1-score

---

## Project Structure

```
├── data/
│   └── transactions.csv         # Kaggle dataset (not tracked)
├── src/
│   ├── preprocess.py            # Feature engineering and scaling
│   ├── dataset.py               # PyTorch Dataset class
│   ├── model.py                 # Autoencoder architecture
│   ├── train.py                 # Training loop
│   ├── evaluate.py              # Metrics and threshold selection
│   └── utils.py                 # Helper functions
├── notebooks/
│   ├── eda.ipynb                # Exploratory data analysis
│   └── results.ipynb            # Result visualization
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/fraud-detection-autoencoder.git
cd fraud-detection-autoencoder
pip install -r requirements.txt
```

---

## Usage

```bash
# Preprocess the data
python src/preprocess.py

# Train the autoencoder
python src/train.py

# Evaluate on the test set
python src/evaluate.py
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| AUROC | > 0.90 |
| AUPRC | > 0.40 |
| Recall | > 0.75 |
| F1-Score | Maximized at chosen threshold |

> Accuracy is not used as a metric due to severe class imbalance.

---

## Tech Stack

- **Python 3.10+**
- **PyTorch** — model definition and training
- **scikit-learn** — preprocessing and evaluation metrics
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualization

---

## Authors

Built as part of the PAAI (Practical Applications of AI) university course.
