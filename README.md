# FidelFolio Market Cap Growth Forecasting

##  Project Overview
This project tackles the **FidelFolio Deep Learning Challenge**, which involves predicting market capitalization growth across multiple time horizons (1Y, 2Y, 3Y) for Indian companies based on fundamental financial indicators using deep learning models.

---

##  Objective
- Model complex, nonlinear relationships between financial indicators and future market cap.
- Predict growth across **short-term (Target 1)**, **medium-term (Target 2)**, and **long-term (Target 3)**.
- Analyze effectiveness of various model architectures (MLP, LSTM, LSTM + Attention, Transformer).
- Use explainability tools (SHAP, LIME) for model interpretability.

---

##  Dataset
- **Rows**: Company-Year instances  
- **Features**: `Feature1` to `Feature28` (fundamental indicators)  
- **Targets**: `Target 1`, `Target 2`, `Target 3` (market cap growth)

### Preprocessing:
- Missing values imputed using company-wise and global means.
- Winsorization to cap outliers.
- Features standardized using `StandardScaler`.

---

##  Models Implemented

### 1. **Multilayer Perceptron (MLP)**
- Feedforward NN with dropout & ReLU
- Trained for 1000 epochs
- RMSE:
```
Target 1: 106.62
Target 2: 227.66
Target 3: 370.31
```

### 2. **LSTM (Vanilla)**
- Sequence-aware model with final timestep target
- Best at 2000 epochs:
```
Target 1: 22.70
Target 2: 56.97
Target 3: 186.78
```

### 3. **LSTM with Attention**
- Uses soft attention for weighted feature aggregation
- Best at 3000 epochs:
```
Target 1: 19.44
Target 2: 47.91
Target 3: 175.11
```

### 4. **Transformer Encoder**
- Positional encoding + multi-head self-attention
- Best at 1200 epochs:
```
Target 1: 30.98
Target 2: 38.77
Target 3: 112.21
```

---

##  Explainability

###  SHAP
- Applied on MLP model using `DeepExplainer`
- Visualizes feature impact on prediction

###  LIME
- Applied on both MLP and LSTM models
- For LSTM, sequences flattened and visualized for interpretability

---

##  Visuals
- Loss curves over epochs for all models
- SHAP summary plots for feature importance
- LIME explanations for per-instance interpretability

_(Include plots in `images/` and reference them in markdown)_

---

##  Model Comparison
| Model              | Attention | RMSE T1 | RMSE T2 | RMSE T3 |
|--------------------|-----------|---------|---------|---------|
| MLP                | No        | 106.62  | 227.66  | 370.31  |
| LSTM               | No        | 22.70   | 56.97   | 186.78  |
| LSTM + Attention   | Yes       | 19.44   | 47.91   | 175.11  |
| Transformer        | Yes       | 30.98   | 38.77   | 112.21  |

---

##  Future Work
- Try Transformer Decoder + Time2Vec
- Incorporate macroeconomic indicators
- Apply hyperparameter tuning with Optuna or Ray Tune
- Explore multi-task learning loss combinations

---

##  Run Instructions
```bash
# Install requirements
pip install -r requirements.txt

# Run MLP training
python train_mlp.py

# Run LSTM
python train_lstm.py

# Run Attention LSTM
python train_lstm_attention.py

# Run Transformer
python train_transformer.py
```

---




