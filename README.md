# NN‑kNN + Tree Ensemble Layer for Financial Forecasting

Hybrid, interpretable multi‑horizon forecasting for financial and economic time series using Neural Network k‑NN and differentiable tree ensembles.

---

## Problem

Financial and economic time series (e.g., prices, indices, macro indicators) are noisy, nonlinear, and require accurate multi‑step‑ahead forecasts where small errors can have large economic impact. 
Classical models (ARIMA, k‑NN, decision trees) are easy to interpret but often struggle on complex patterns, while deep learning models (e.g., LSTMs) can be accurate but behave as black boxes, which is problematic in finance where decisions must be justified. 

---

## Our Solution

This project builds a forecasting system that aims to keep **deep‑learning‑level accuracy** while preserving **k‑NN / tree‑style interpretability**:

- **NN‑kNN (Neural Network k‑Nearest Neighbors)**  
  A neural architecture that compares the current time window to stored historical windows and bases predictions on their labels, while learning the similarity metric and weights end‑to‑end. 
  It retains neighbor‑based explanations (“this forecast is influenced by these specific past periods”). 

- **Differentiable Tree Ensemble Layer**  
  A neural decision forest layer with soft, probabilistic splits and trainable leaf predictions, giving rule‑like explanations and feature importance while being trained with backpropagation. 

- **Multi‑horizon forecasting**  
  The model outputs 1‑, 5‑, and 20‑step‑ahead forecasts in a shared architecture to capture dependencies across horizons. 

---

## What We’re Trying to Do

We focus on two core research questions:

1. Can NN‑kNN match standard deep models (e.g., an LSTM baseline) on financial time‑series forecasting **while preserving neighbor‑based interpretability**? 
2. Does adding a differentiable Tree Ensemble Layer **improve interpretability** (rules, feature importance) **without significantly hurting accuracy** on benchmarks like the financial subset of M4 and the ETT dataset? 

Evaluation uses SMAPE, MASE, RMSE, MAE, and directional accuracy, plus interpretability measures such as neighbor similarity and rule stability.

---

## Tech Stack

- **Python 3.9+**
- **PyTorch** – NN‑kNN, LSTM encoder, Tree Ensemble Layer. 
- **scikit‑learn** – Classical k‑NN, decision trees, random forests, PCA, Fisher’s LDA 
- **statsmodels / pmdarima** – ARIMA baselines. 
- **pandas**, **numpy** – Data handling.  
- **Matplotlib** – Visualization.  
