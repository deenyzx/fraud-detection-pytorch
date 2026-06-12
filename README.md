# fraud-detection-pytorch

A PyTorch-based anomaly detection system that identifies fraudulent credit card transactions using an autoencoder neural network.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project frames fraud detection as an **unsupervised anomaly detection** problem. The autoencoder is trained exclusively on legitimate transactions and learns to reconstruct normal behavior. Fraudulent transactions, being structurally different, produce a high reconstruction error — which is used as the anomaly score.

---

## Dataset

[Kaggle Credit Card Transactions Dataset](https://www.kaggle.com/datasets/tjverry/credit-card-transactions)

- 284,807 transactions
- 492 fraud cases (~0.17% of all transactions)
- Features used: `transactionamount`, `availablemoney`, `currentbalance`, `creditlimit`, `cardpresent`, `cvv_match` (engineered)

> Download the dataset from Kaggle and place the zip file at `data/creditcard.zip` before running.

---

## How It Works

1. **Feature Engineering** — A `cvv_match` feature is derived by comparing `cardcvv` vs `enteredcvv`
2. **Preprocessing** — Continuous features are standardized using `StandardScaler`; the scaler is saved for inference
3. **Training** — The autoencoder is trained **only on normal transactions**, learning to reconstruct legitimate behavior
4. **Inference** — Reconstruction error (MSE) is computed per transaction; high error signals an anomaly
5. **Thresholding** — The optimal threshold is selected by maximizing F1-score on the precision-recall curve
6. **Evaluation** — Model is assessed using PR-AUC, Precision, Recall, and F1-score

---

## Project Structure

```
fraud-detection-pytorch/
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Data ingestion, feature engineering, preprocessing
│   ├── model.py              # Autoencoder architecture
│   ├── training.py           # Training loop
│   └── evaluation.py         # Metrics and threshold selection
├── data/
│   └── creditcard.zip        # Dataset (download from Kaggle)
├── models/                   # Auto-created on first run
│   ├── autoencoder_fraud.pth # Saved model weights
│   ├── scaler.pkl            # Saved StandardScaler for inference
│   └── threshold.json        # Optimal decision threshold
├── main.py                   # Full training pipeline
├── run.ipynb                 # End-to-end Colab notebook (train + evaluate + inference)
├── config.yaml               # Hyperparameters and paths
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/deenyzx/fraud-detection-pytorch.git
cd fraud-detection-pytorch
pip install -r requirements.txt
```

---

## Usage

### Option A — Colab (recommended)

Open `run.ipynb` in Google Colab and run all cells top to bottom. The notebook handles everything: training, evaluation charts, and a live interactive inference form where you can manually enter a transaction and receive a fraud risk score.

### Option B — Local

```bash
# Train the model, save weights + scaler + threshold
python main.py

# Then open run.ipynb in Jupyter for evaluation charts and inference
```

---

## Results

| Metric | Value |
|--------|-------|
| PR-AUC | 0.1720 |
| Precision | 0.1557 |
| Recall | 0.6964 |
| F1-Score | 0.2546 |

---

## Tech Stack

- **Python 3.10+**
- **PyTorch** — model definition and training
- **scikit-learn** — preprocessing and evaluation metrics
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualization
- **ipywidgets** — interactive inference form in Colab

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## License

[MIT](https://choosealicense.com/licenses/mit/)

---

*Built as part of the PAAI (Practical Applications of AI) university course.*
