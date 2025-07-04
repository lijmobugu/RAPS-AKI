
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import catboost

# Load model
try:
    model = joblib.load('final_stacking_model.pkl')
except (FileNotFoundError, AttributeError, ModuleNotFoundError) as e:
    st.error(f"Model loading failed: {e}")
    st.info("Model file is incompatible with current environment. Please retrain the model or check dependency versions.")
    model = None
    st.stop()

# Feature range definitions
feature_names = [
    "Age", "Diabetes", "AST/ALT(DRR)", "Creatinine (Cr)", "INR", "PT", 
    "Estimated Blood Loss (EBL) > 300 mL", "eGFR", "Tumor Dimension (mm)", 
    "Intraoperative Complications"
]

feature_ranges = {
    "Age": {"type": "numerical", "min": 0, "max": 200, "default": 50},
    "Diabetes": {"type": "categorical", "options": ["YES", "NO"]},
    "AST/ALT(DRR)": {"type": "numerical", "min": 0, "max": 10, "default": 1.0},
    "Creatinine (Cr)": {"type": "numerical", "min": 0, "max": 10, "default": 1.0},
    "INR": {"type": "numerical", "min": 0.5, "max": 5.0, "default": 1.0},
    "PT": {"type": "numerical", "min": 10, "max": 50, "default": 12},
    "Estimated Blood Loss (EBL) > 300 mL": {"type": "categorical", "options": ["YES", "NO"]},
    "eGFR": {"type": "numerical", "min": 0, "max": 200, "default": 90},
    "Tumor Dimension (mm)": {"type": "numerical", "min": 0, "max": 200, "default": 30},
    "Intraoperative Complications": {"type": "categorical", "options": ["YES", "NO"]}
}

# Streamlit interface
st.title("🏥 AKI Prediction Model")
st.header("Please enter the following clinical parameters:")

# Create two-column layout
col1, col2 = st.columns(2)

feature_values = {}
for i, (feature, properties) in enumerate(feature_ranges.items()):
    # Alternate placement between two columns
    current_col = col1 if i % 2 == 0 else col2
    
    with current_col:
        if properties["type"] == "numerical":
            feature_values[feature] = st.number_input(
                label=f"{feature}",
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
                help=f"Range: {properties['min']} - {properties['max']}"
            )
        elif properties["type"] == "categorical":
            feature_values[feature] = st.selectbox(
                label=f"{feature}",
                options=properties["options"],
            )

# Process categorical features
processed_values = feature_values.copy()
label_encoders = {}

for feature, properties in feature_ranges.items():
    if properties["type"] == "categorical":
        label_encoders[feature] = LabelEncoder()
        label_encoders[feature].fit(properties["options"])
        processed_values[feature] = label_encoders[feature].transform([feature_values[feature]])[0]

# Convert to model input format
features = pd.DataFrame([processed_values], columns=feature_names)

# Prediction functionality (without SHAP)
if st.button("🔍 Run Prediction", type="primary"):
    try:
        # Model prediction
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]

        # Display prediction results
        st.subheader("📊 Prediction Results:")
        
        # Create results display
        col1, col2 = st.columns(2)
        
        with col1:
            if predicted_class == 1:
                st.error(f"⚠️ Prediction: High Risk")
                st.error(f"AKI Probability: **{predicted_proba[1]*100:.1f}%**")
            else:
                st.success(f"✅ Prediction: Low Risk")
                st.success(f"AKI Probability: **{predicted_proba[1]*100:.1f}%**")
        
        with col2:
            # Display probability distribution
            prob_data = pd.DataFrame({
                'Risk Category': ['Low Risk', 'High Risk'],
                'Probability': [predicted_proba[0]*100, predicted_proba[1]*100]
            })
            st.bar_chart(prob_data.set_index('Risk Category'))
        
        # Detailed probability information
        st.subheader("📋 Detailed Prediction Information:")
        
        # Create probability table
        prob_df = pd.DataFrame({
            'Risk Category': ['Low Risk (Class 0)', 'High Risk (Class 1)'],
            'Predicted Probability': [f"{predicted_proba[0]*100:.2f}%", f"{predicted_proba[1]*100:.2f}%"],
            'Confidence Score': [f"{predicted_proba[0]:.4f}", f"{predicted_proba[1]:.4f}"]
        })
        
        st.dataframe(prob_df, use_container_width=True)
        
        # Risk interpretation
        st.subheader("🔍 Clinical Interpretation:")
        
        if predicted_class == 1:
            st.warning("""
            **High Risk Prediction:**
            - The model predicts a high probability of AKI development for this patient
            - Enhanced monitoring and preventive measures are recommended
            - Please consider clinical context and additional risk factors in decision-making
            """)
        else:
            st.info("""
            **Low Risk Prediction:**
            - The model predicts a low probability of AKI development for this patient
            - Standard monitoring protocols should be maintained
            - Please consider clinical context and additional risk factors in decision-making
            """)
        
        # Input feature review
        st.subheader("📝 Input Parameter Summary:")
        
        # Create feature table
        feature_df = pd.DataFrame({
            'Clinical Parameter': feature_names,
            'Input Value': [feature_values[name] for name in feature_names],
            'Data Type': [feature_ranges[name]['type'] for name in feature_names]
        })
        
        st.dataframe(feature_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {e}")
        st.info("Please verify input values are correct or contact the system administrator.")

# Add footer information
st.markdown("---")
st.markdown("*This prediction model is for medical research purposes only and should not replace professional clinical judgment*")
