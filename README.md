# Credit Scoring with Neural Network

A deep learning project focused on credit scoring and predicting loan defaults using modern tabular data preprocessing pipelines and optimized Deep Neural Networks (DNN) in Keras/TensorFlow.

## 📊 Business Problem & Objectives
The goal is to accurately rank loan applicants based on their probability of default (`LoanPayBack`). 
- **Class 0:** Default / High Risk (Bank loses money)
- **Class 1:** Paid Back / Reliable (Bank generates profit)

Given the extreme class imbalance (~80% vs ~20%), standard accuracy is insufficient. The main technical metric optimized during development is **ROC AUC**, while the operational threshold is tuned using business risk management constraints.

## 🛠️ Tech Stack & Key Features
- **Data Manipulation:** Python, Pandas, NumPy
- **Preprocessing & Pipelines:** `scikit-learn` (`ColumnTransformer`, `OneHotEncoder`, `StandardScaler`)
- **Deep Learning Framework:** `Keras` / `TensorFlow`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Optimization Tech:** Learning Rate Decay, Custom Class Weights, Dynamic Threshold Tuning (Youden's J Statistic)

## 🧠 Model Architecture & Training Optimization
To avoid gradient collapse and over-parameterization on noisy financial tabular data, the following custom DNN structure was implemented:
- **Input Layer:** Dynamically shapes to preprocessed features after categorical encoding.
- **Dense Layers:** 256 ➡️ 128 ➡️ 32 neurons.
- **Stability:** `BatchNormalization` applied directly *before* `ReLU` activations to maintain healthy gradient flow.
- **Regularization:** Heavy `Dropout` (up to 0.4) to mitigate overfitting.
- **Output Layer:** Single neuron with `sigmoid` activation spitting out continuous probabilities $[0.0, 1.0]$.
- **Optimizer:** `Adam` with a highly stabilized learning rate ($1e-5$).

## 📈 Final Evaluation & Metrics

### ROC Curve & Classification Performance
The model exhibits exceptional stability, converging cleanly without jumping validation losses:
- **Test ROC AUC:** `~0.865`
- **Overall Accuracy:** `86%`


## Model Deployment

Download the pre-trained model to make predictions without retraining:

* [Download Pre-trained Model (model_weights.keras)](https://drive.google.com/file/d/1W-gsf-FTzNVT9vZ47NCccdl8H8L6QdoD/view?usp=drive_link)

```python
from keras.models import load_model

model = load_model('creditscoringnn_model_weights.keras')
```
