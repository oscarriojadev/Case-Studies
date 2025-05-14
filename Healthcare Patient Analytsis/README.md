# 🏥 Healthcare Patient Readmission Analysis

This Python script analyzes healthcare patient data to predict hospital readmissions using a Random Forest Classifier. It handles data preprocessing, training, evaluation, and visualization of model performance and feature importance.

---

## 📂 File Structure

* `healthcare_patient_analysis.py`: Main script for loading, preprocessing, training, evaluating, and visualizing patient readmission prediction.
* `path_to_your_data.csv`: CSV file containing your patient dataset (you must provide this file).

---

## 🧠 Objective

Predict whether a patient will be readmitted to the hospital based on clinical and demographic features.

---

## 🛠️ Features

* 📦 **Data Loading** via Pandas
* 🧹 **Preprocessing** with feature scaling
* ✂️ **Train/Test Splitting** using `train_test_split`
* 🌲 **Model Training** with `RandomForestClassifier`
* 📊 **Evaluation**: Accuracy, Precision, Recall, F1, Confusion Matrix
* 🔎 **Feature Importance** visualization
* 📈 **Classification Report** output

---

## 🚀 How to Use

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Add Your Dataset

Replace `'path_to_your_data.csv'` with the actual path to your CSV file in the `main()` function.

### 3. Set Target Column

Make sure your dataset includes a target column (e.g., `Readmission`) with binary classification (0 = No, 1 = Yes).

### 4. Run the Script

```bash
python healthcare_patient_analysis.py
```

---

## 📋 Expected Columns in Dataset

* Clinical and demographic features (e.g., age, diagnosis codes, lab results)
* Target column (e.g., `Readmission`)

---

## 📊 Outputs

* ✅ Accuracy, Precision, Recall, F1 Score
* 🧾 Classification Report
* 🔁 Confusion Matrix Heatmap
* 🧠 Feature Importance Bar Chart

---

## 📌 Notes

* Adjust `RandomForestClassifier()` parameters for better performance.
* For multiclass or regression problems, modify the model and evaluation metrics accordingly.

---

## 👨‍⚕️ Example Use Case

Predict whether a diabetic patient will be readmitted within 30 days after discharge based on their medical history, treatments, and demographics.
