# Diabetes Prediction App

A Streamlit web application for predicting the likelihood of diabetes using health parameters from the Pima Indians Diabetes dataset.

## Features
- **Title & Description**: Clear header and purpose.
- **Sidebar Navigation**: Switch between Data Exploration, Visualizations, Model Prediction, Model Performance.
- **Data Exploration**: Shape, column types, sample data, and interactive filters.
- **Visualizations**: Outcome distribution, Glucose vs Age, correlation heatmap, and custom scatter.
- **Model Prediction**: Input widgets for all features with real-time prediction and probability.
- **Model Performance**: Cross-validation metrics, best model, hold-out metrics, and confusion matrix.
- **Technical**: Error handling, loading spinners, consistent layout, help text.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Retrain Model (optional)
```bash
python train_model.py
```

## Project Structure
```
diabetes-app/
├─ app.py
├─ train_model.py
├─ requirements.txt
├─ model/
│  ├─ diabetes_model.pkl
│  └─ metrics.json
├─ data/
│  └─ diabetes.csv
├─ notebooks/
└─ README.md
```
