# ✈️ Flight Delay Prediction

This Python script builds a machine learning model to predict whether a flight will be delayed, using a dataset with various flight-related features.

## 🔍 Overview

The pipeline includes:

* 📥 Data loading (`.csv`)
* 🧹 Preprocessing and feature scaling
* 🤖 Training a **Random Forest Classifier**
* 📊 Evaluation with accuracy, precision, recall, and F1-score
* 🔍 Confusion matrix & feature importance visualization

## 📁 File Structure

* `flight_delay_prediction.py`: Main script containing all functions and the `main()` execution block
* `README.md`: You're reading it 😉

## 🛠️ Requirements

Install the required packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## 📈 Features Used

* **amount**: Transaction amount
* **time**: Transaction time
* **feature1**: Risk score
* **feature2**: Account age
* **feature3**: Location risk
* **feature4**: Transaction velocity
* `Delay`: **Target** binary column (0 = No delay, 1 = Delay)

> ⚠️ Replace `path_to_your_data.csv` with the actual path to your dataset

# 🧠 Why Random Forest?

The model used in this project is a **Random Forest Classifier**, chosen for the following reasons:

### ✅ Robustness to Overfitting
Random Forest aggregates results from multiple decision trees, reducing overfitting and improving generalization compared to a single tree model.

### ✅ Handles Mixed Data Types
It performs well with both numerical and categorical features, making it ideal for heterogeneous flight-related datasets.

### ✅ Feature Importance Insight
It provides built-in mechanisms to evaluate feature importance, helping interpret what factors contribute most to flight delays.

### ✅ High Accuracy on Classification Tasks
Random Forests are known for strong performance in binary classification tasks like "Delay" vs. "No Delay", often outperforming simpler models like Logistic Regression without requiring complex tuning.

### ✅ Minimal Preprocessing Required
Unlike many linear models, Random Forests don't require feature scaling or normalization, streamlining the preprocessing pipeline.

---

## 🔍 Model Justification: Why Random Forest?

To determine the best model for predicting flight delays, we evaluated several popular classification algorithms based on performance, interpretability, preprocessing needs, and robustness.

| **Model**              | **Accuracy**   | **Overfitting Risk** | **Preprocessing Required** | **Feature Importance** | **Interpretability** | **Comments**                                                                 |
|------------------------|----------------|----------------------|----------------------------|------------------------|----------------------|-----------------------------------------------------------------------------|
| **Random Forest**      | ✅ High        | 🔒 Low               | ⚙️ Minimal                | ✅ Yes                  | 🟡 Moderate           | Strong balance of performance & simplicity                                     |
| **Logistic Regression**| 🟡 Medium      | 🟢 Very Low          | ⚙️ Scaling required       | ⚠️ Limited             | ✅ High               | Great baseline, but underperforms on non-linear data                        |
| **Support Vector Machine**| ✅ High      | ⚠️ Medium            | ⚙️ Scaling required       | ❌ No                  | 🔴 Low                | Performs well but hard to interpret; slower with large datasets              |
| **XGBoost**            | ✅ Very High   | 🟡 Medium            | ⚙️ Minimal                | ✅ Yes                  | 🟡 Moderate           | Top-tier performance but more complex tuning and dependencies               |
| **K-Nearest Neighbors**| 🟡 Medium      | ✅ Low               | ⚙️ Scaling required       | ❌ No                  | 🔴 Low                | Simple but inefficient for large datasets                                    |

---

## 🎯 Summary

Given the dataset characteristics (mixture of features, need for interpretability, and performance), **Random Forest** offers:

- Excellent accuracy without complex tuning
- Resistance to overfitting via ensemble learning
- Insights into which features matter most
- A relatively easy-to-use model with good defaults

For future optimization, trying **XGBoost** may offer a performance boost in exchange for added complexity.



## ▶️ How to Run

```bash
python flight_delay_prediction.py
```

## 📊 Model Output

* Performance metrics: Accuracy, Precision, Recall, F1
* Confusion matrix heatmap
* Feature importance plot
* Classification report

## 📌 Key Functions

* `load_data()`: Loads CSV data
* `preprocess_data()`: Scales features and splits target
* `split_data()`: Train-test split
* `train_model()`: Trains RandomForestClassifier
* `evaluate_model()`: Outputs main performance metrics
* `plot_confusion_matrix()`: Displays confusion matrix
* `plot_feature_importance()`: Shows which features matter most

## 📬 License

MIT License — use freely with attribution.
