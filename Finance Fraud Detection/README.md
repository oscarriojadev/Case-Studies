# 💰 Financial Fraud Detection with Random Forest

![Machine Learning](https://img.shields.io/badge/ML-Random_Forest-green)
![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A machine learning solution for detecting fraudulent financial transactions using **Random Forest** classification.

---

## 🚀 Features

- 🔍 Data preprocessing and standardization  
- ✂️ Train-test splitting  
- 🌲 Random Forest classifier implementation  
- 📊 Model evaluation metrics:
  - ✅ Accuracy
  - 🎯 Precision
  - 🔁 Recall
  - ⚖️ F1-score
- 📈 Visualization tools:
  - 🧮 Confusion matrix
  - 🌟 Feature importance plot
- 🧾 Detailed classification report

---

## 📦 Requirements

- Python 3.7+
- Install dependencies with:
  ```bash
  pip install -r requirements.txt
   ```


**Required packages**:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## Methodology
## Justification for Using Random Forest

### Advantages of Random Forest

#### Handling High-Dimensional Data

**Description:** Efficiently handles datasets with many features.

**Justification:** Financial datasets often contain numerous variables. Random Forest's ability to handle this makes it suitable for fraud detection.

---

#### Robustness to Overfitting

**Description:** Aggregates multiple decision trees to reduce variance and improve generalization.

**Justification:** Ensures good performance on unseen data by minimizing overfitting risks.

---

#### Feature Importance

**Description:** Offers straightforward assessment of which features are most influential.

**Justification:** Enhances interpretability and supports informed decision-making.

---

#### Handling Non-Linear Relationships

**Description:** Captures non-linear patterns without explicit feature engineering.

**Justification:** Many financial relationships are non-linear; Random Forest naturally models these complexities.

---

#### Robustness to Outliers and Noise

**Description:** Performs well even with imperfect data.

**Justification:** Ensures model reliability in real-world, noisy datasets.

---

#### Ease of Use

**Description:** Easy to implement and requires fewer hyperparameters compared to other models.

**Justification:** Facilitates quick deployment and tuning, ideal for fast-paced environments.

---

## Comparison with Other Models

### Logistic Regression

- **Pros:** Simple, interpretable, efficient for linearly separable data.
- **Cons:** Struggles with non-linear relationships and high-dimensional data; less robust to outliers.

---

### Support Vector Machines (SVM)

- **Pros:** Effective in high-dimensional spaces; flexible with kernel functions.
- **Cons:** Computationally intensive with large datasets; requires careful tuning.

---

### Neural Networks

- **Pros:** Highly flexible; models complex relationships.
- **Cons:** Requires large data and computational resources; can be hard to interpret; prone to overfitting.

---

### Gradient Boosting Machines (GBM)

- **Pros:** High predictive accuracy; handles non-linear relationships.
- **Cons:** More prone to overfitting than Random Forest; requires careful tuning; can be resource-intensive.

---

## 🛠️ Usage

1. Dataset:

   Dataset Description:
    *amount: Transaction amount (higher amounts more likely to be fraudulent)
    
    *time: Transaction time (fraud often occurs at unusual hours)
    
    *feature1: Normalized transaction frequency (higher for fraud)
    
    *feature2: Account age in days (newer accounts more likely to be fraudulent)
    
    *feature3: Location risk score (higher for fraud)
    
    *feature4: Transaction velocity (higher for fraud)
    
    *Fraud: Target variable (1 = fraudulent, 0 = legitimate)
    
  Characteristics:
    *15 sample transactions (10 legitimate, 5 fraudulent)
    *
    *Fraudulent transactions tend to have:
    *
    *Higher amounts (> $1800)
    *
    *Higher feature values (feature1, feature3, feature4)
    *
    *Occur at unusual hours (midnight to 5am)
    *
    *Legitimate transactions are smaller and occur during normal business hours
    *
    *The dataset is designed to show clear patterns that a Random Forest classifier should be able to detect.


2. Run the model:

   ```bash
   python finance_fraud_detection.py
   ```

---

## 🧠 Model Workflow

1. **Data Loading**: Reads the CSV data
2. **Preprocessing**:

   * Separates features & target
   * Applies `StandardScaler` to standardize
3. **Train-Test Split** (80/20)
4. **Model Training** using `RandomForestClassifier`
5. **Evaluation**:

   * Computes metrics
   * Plots confusion matrix
   * Shows feature importances

---

## 📤 Example Output

```python
Accuracy: 0.9925, Precision: 0.8765, Recall: 0.8120, F1: 0.8429

              precision    recall  f1-score   support

           0       0.99      1.00      0.99      5686
           1       0.88      0.81      0.84        99

    accuracy                           0.99      5785
   macro avg       0.93      0.91      0.92      5785
weighted avg       0.99      0.99      0.99      5785
```

---

## ⚙️ Customization Options

Adjust parameters in `main()`:

```python
file_path = 'path_to_your_data.csv'       # Your dataset
target_column = 'Fraud'                   # Target variable
test_size = 0.2                           # Test set split ratio
random_state = 42                         # For reproducibility
```

---

## 📐 Performance Metrics

* 🎯 **Precision**: Minimize false positives
* 🚨 **Recall**: Maximize fraud detection
* ⚖️ **F1-score**: Balance both precision and recall

---

## 📊 Visualizations

1. **Confusion Matrix** – Overview of prediction accuracy
2. **Feature Importance Plot** – See what influences fraud detection

---

## 🪪 License

MIT License.

---

## 🤝 Contributing

Contributions are welcome!
Feel free to open issues or PRs to:

* Add new models (e.g., XGBoost, Neural Networks)
* Improve feature engineering
* Enhance performance
