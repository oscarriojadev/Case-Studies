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

> ⚠️ Replace `path_to_your_data.csv` with the actual path to your dataset.

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

---

Let me know if you want a version with badges or formatted for Jupyter Notebooks.
