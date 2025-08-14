import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

DATA_PATH = "data/diabetes.csv"
MODEL_PATH = "model/diabetes_model.pkl"
METRICS_PATH = "model/metrics.json"

# ---------- Helpers (cached) ----------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        pipe = pickle.load(f)
    return pipe

@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)

# ---------- UI Header ----------
st.title("🩺 Diabetes Prediction App")
st.caption("Predict the likelihood of diabetes using a trained ML model. Explore the data, visualize trends, and make real-time predictions.")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    options=["Data Exploration", "Visualizations", "Model Prediction", "Model Performance"],
    help="Use this menu to navigate between sections of the app."
)

# Load resources with loading indicators
with st.spinner("Loading dataset..."):
    df = load_data()
with st.spinner("Loading model..."):
    pipe = load_model()
with st.spinner("Loading metrics..."):
    metrics = load_metrics()

# ---------- Pages ----------
if page == "Data Exploration":
    st.header("📊 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", int(df.isna().sum().sum()))

    st.markdown("**Column Types**")
    st.dataframe(pd.DataFrame({"dtype": df.dtypes.astype(str)}))

    st.markdown("**Sample (first 20 rows)**")
    st.dataframe(df.head(20))

    st.markdown("---")
    st.subheader("🔎 Interactive Filtering")
    st.caption("Use sliders to filter numeric columns by ranges.")
    selected = st.multiselect("Choose columns to filter by:", options=df.columns.tolist())
    filtered = df.copy()
    for col in selected:
        if np.issubdtype(df[col].dtype, np.number):
            low, high = float(df[col].min()), float(df[col].max())
            val = st.slider(f"{col} range", min_value=low, max_value=high, value=(low, high))
            filtered = filtered[(filtered[col] >= val[0]) & (filtered[col] <= val[1])]
        else:
            st.info(f"Skipping **{col}** (non-numeric).")
    st.success(f"Filtered rows: {filtered.shape[0]}")
    st.dataframe(filtered.head(50))

elif page == "Visualizations":
    st.header("📈 Visualizations")
    st.caption("Interactive charts to explore relationships in the dataset.")
    # Chart 1: Outcome distribution
    if "Outcome" in df.columns:
        st.subheader("Outcome Distribution")
        fig1 = px.histogram(df, x="Outcome", nbins=2, title="Outcome (0/1)")
        st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Glucose vs Age by Outcome
    if all(c in df.columns for c in ["Glucose", "Age"]):
        st.subheader("Glucose vs. Age (colored by Outcome)")
        color = df["Outcome"].astype(str) if "Outcome" in df.columns else None
        fig2 = px.scatter(df, x="Age", y="Glucose", color=color, title="Glucose vs Age")
        st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig3 = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlations")
    st.plotly_chart(fig3, use_container_width=True)

    # Bonus: user-selectable pairplot-like scatter
    st.subheader("Custom Scatter")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    x_axis = st.selectbox("X-axis", options=num_cols, index=0)
    y_axis = st.selectbox("Y-axis", options=num_cols, index=1 if len(num_cols) > 1 else 0)
    fig4 = px.scatter(df, x=x_axis, y=y_axis, color=df["Outcome"].astype(str) if "Outcome" in df.columns else None,
                      title=f"{x_axis} vs {y_axis}")
    st.plotly_chart(fig4, use_container_width=True)

elif page == "Model Prediction":
    st.header("🧮 Model Prediction")
    st.caption("Enter feature values to get a prediction and probability.")

    # Build input widgets dynamically based on columns except target
    feature_cols = [c for c in df.columns if c != "Outcome"]
    cols = st.columns(4)
    user_inputs = {}
    for i, feat in enumerate(feature_cols):
        with cols[i % 4]:
            if np.issubdtype(df[feat].dtype, np.number):
                default = float(df[feat].median())
                user_inputs[feat] = st.number_input(feat, value=default, help=f"Enter a numeric value for {feat}")
            else:
                user_inputs[feat] = st.text_input(feat, value=str(df[feat].mode()[0]), help=f"Enter a value for {feat}")

    # Predict button
    if st.button("Predict", help="Run the trained model on the values you entered."):
        try:
            X_input = pd.DataFrame([user_inputs], columns=feature_cols)
            with st.spinner("Running prediction..."):
                pred = int(pipe.predict(X_input)[0])
                proba = pipe.predict_proba(X_input)[0][1] if hasattr(pipe, "predict_proba") else None
            st.success(f"Prediction: **{pred}**")
            if proba is not None:
                st.info(f"Probability of diabetes (class 1): **{proba:.3f}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif page == "Model Performance":
    st.header("📏 Model Performance")
    st.caption("Evaluation metrics, confusion matrix, and model comparison results.")

    # Show cross-validation results & best model
    st.subheader("Cross-Validation (ROC AUC)")
    st.json(metrics["cv_results"])
    st.write(f"**Best Model:** {metrics['best_model']}")

    # Hold-out metrics
    st.subheader("Hold-out Metrics (20% split)")
    st.write({"accuracy": metrics["holdout_accuracy"], "roc_auc": metrics["holdout_roc_auc"]})

    # Confusion matrix from a re-created split (for display)
    if "Outcome" in df.columns:
        X = df.drop(columns=["Outcome"])
        y = df["Outcome"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        y_pred = pipe.predict(X_te)
        cm = confusion_matrix(y_te, y_pred)
        st.subheader("Confusion Matrix")
        st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

# Footer help
st.markdown("---")
with st.expander("ℹ️ Help & Documentation"):
    st.write("""
    **How to use this app**
    - Use the sidebar to move between sections.
    - In **Data Exploration**, inspect the dataset, types, and filter rows interactively.
    - In **Visualizations**, explore distributions and relationships.
    - In **Model Prediction**, enter values to compute a prediction and (if supported) probability.
    - In **Model Performance**, see metrics and confusion matrix.
    """)
