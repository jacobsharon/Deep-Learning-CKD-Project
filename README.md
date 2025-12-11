# CKD_Project

## Chronic Kidney Disease Classification Using Multi-Layer Perceptrons (MLPs)

This project implements a complete CKD classification workflow using PyTorch inside a Jupyter Notebook. The notebook loads the dataset directly from the UCI Machine Learning Repository using `ucimlrepo`, preprocesses all features, trains MLP models, evaluates their performance, and generates SHAP interpretability plots.

---

## Dataset Source

The dataset is imported automatically:

- Source: UCI Machine Learning Repository  
- Dataset: Chronic Kidney Disease (ID 336)  
- Method: `fetch_ucirepo` (no manual download required)

The target variable is mapped as:
- `ckd` → 1  
- `notckd` → 0  

---

## Environment Setup

Install all required dependencies:

### Dataset Import
- ucimlrepo

### Data / Preprocessing
- pandas  
- numpy  
- sklearn.impute.KNNImputer  
- sklearn.preprocessing.MinMaxScaler  
- sklearn.model_selection.train_test_split  
- sklearn.model_selection.StratifiedKFold  

### PyTorch (Model + Training)
- torch  
- torch.nn  
- torch.nn.init  
- torch.utils.data.TensorDataset  
- torch.utils.data.DataLoader  

### Metrics
- accuracy_score  
- precision_score  
- recall_score  
- confusion_matrix  
- roc_auc_score  
- roc_curve  

### Plotting
- matplotlib.pyplot  

### Explainability
- shap  

### Utilities
- os  
- itertools  
- datetime  

---

Then run the full workflow:

**Kernel → Restart & Run All**

---

## Notebook Workflow

### 1. Data Loading
- Loads CKD dataset using `fetch_ucirepo(id=336)`
- Extracts feature matrix (`X`) and label vector (`y`)
- Cleans inconsistent string formatting

### 2. Preprocessing
The notebook performs:
- Dropping of non-informative identifier fields  
- Standardization of categorical string values  
- Mapping categorical and binary values (e.g., yes/no → 1/0)  
- Separation of numerical, categorical, and binary feature types  
- Missing value imputation using **KNNImputer**  
- Feature scaling with **StandardScaler**  
- Casting all features to `float32`  

### 3. Model Architecture
An MLP classifier is defined with:
- Linear layers  
- ReLU activations  
- Batch normalization  
- Dropout  
- Kaiming uniform initialization  
- A single-logit output layer for binary classification  

### 4. Training Configuration
- Optimizer: AdamW  
- Loss: BCEWithLogitsLoss  
- Batch size: 32  
- 70/30 training–validation split  
- Early stopping based on validation loss  
- Epoch-wise training and validation loss tracking  

### 5. Threshold Optimization
A sweep from **0.0 to 1.0** identifies the threshold that maximizes **F1-score**, improving classification accuracy over a default threshold of 0.5.

### 6. Evaluation Metrics
The notebook computes:
- Accuracy  
- Precision  
- Recall (Sensitivity)  
- Specificity  
- F1 Score  
- AUC (ROC)  
- Youden’s J statistic  
- Confusion matrix  

### 7. Visualization Outputs
The notebook generates:
- ROC curve  
- Training and validation loss curves  
- Confusion matrix plot  

### 8. SHAP Interpretability
The notebook produces:
- SHAP summary plot  
- SHAP dependence plots for highly influential clinical features  

These explain how input variables contribute to model predictions.


## Reproducibility Notes

- Models are reinitialized each run  
- SHAP uses 100 background and 200 evaluation samples  
- Notebook cells must be executed sequentially  
- Optional: set seeds for deterministic behavior  

---

## References

UCI CKD Dataset: https://doi.org/10.24432/C5G020  
ucimlrepo: https://pypi.org/project/ucimlrepo/

---

## Author

Jacob Sharon, BS, MLS(ASCP)CM  
Graduate Student, Computer Science  
Missouri State University