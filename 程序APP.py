
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
import shap

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
    "Age": {"type": "numerical", "min": 18, "max": 80, "default": 50},
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

# Create background data for SHAP
@st.cache_data
def create_background_data():
    """Create background dataset for SHAP explanation"""
    # Generate representative background samples
    np.random.seed(42)
    n_samples = 50  # Use fewer samples for faster computation
    
    background_data = []
    for _ in range(n_samples):
        sample = {}
        for feature, properties in feature_ranges.items():
            if properties["type"] == "numerical":
                # Generate random values within the range
                min_val = properties["min"]
                max_val = properties["max"]
                sample[feature] = np.random.uniform(min_val, max_val)
            elif properties["type"] == "categorical":
                # Random choice from options, then encode
                choice = np.random.choice(properties["options"])
                le = LabelEncoder()
                le.fit(properties["options"])
                sample[feature] = le.transform([choice])[0]
        background_data.append(sample)
    
    return pd.DataFrame(background_data, columns=feature_names)

# SHAP explanation functions
@st.cache_resource
def create_shap_explainer(model, background_data):
    """Create SHAP explainer for StackingClassifier"""
    try:
        # Use KernelExplainer for StackingClassifier
        def model_predict(X):
            return model.predict_proba(X)[:, 1]  # Return probability of positive class
        
        explainer = shap.KernelExplainer(model_predict, background_data.values)
        return explainer
    except Exception as e:
        st.error(f"Failed to create SHAP explainer: {e}")
        return None

def generate_shap_explanation(explainer, features, feature_names):
    """Generate SHAP values and create visualizations"""
    try:
        if explainer is None:
            return None, None
        
        # Calculate SHAP values (this may take some time)
        shap_values = explainer.shap_values(features.values, nsamples=100)
        
        # Create SHAP waterfall plot
        fig_waterfall = plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=features.values[0],
                feature_names=feature_names
            ),
            max_display=10,
            show=False
        )
        plt.title("SHAP Waterfall Plot - Feature Contributions")
        plt.tight_layout()
        
        # Create SHAP bar plot
        fig_bar = plt.figure(figsize=(10, 6))
        shap.bar_plot(
            shap.Explanation(
                values=shap_values[0],
                feature_names=feature_names
            ),
            max_display=10,
            show=False
        )
        plt.title("SHAP Bar Plot - Feature Importance")
        plt.tight_layout()
        
        return fig_waterfall, fig_bar, shap_values
        
    except Exception as e:
        st.error(f"Failed to generate SHAP explanation: {e}")
        return None, None, None

# Streamlit interface
st.title("🏥 AKI Prediction Model with SHAP Explanations")
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

# Prediction functionality with SHAP
if st.button("🔍 Run Prediction & Analysis", type="primary"):
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
        
        # SHAP Analysis
        st.subheader("🔍 SHAP Model Explanation:")
        
        # Option to enable/disable SHAP (due to computational cost)
        enable_shap = st.checkbox("Enable SHAP Analysis (may take 30-60 seconds)", value=True)
        
        if enable_shap:
            with st.spinner("Generating SHAP explanations... This may take a moment."):
                # Create background data
                background_data = create_background_data()
                
                # Create SHAP explainer
                explainer = create_shap_explainer(model, background_data)
                
                if explainer is not None:
                    # Generate SHAP explanation
                    fig_waterfall, fig_bar, shap_values = generate_shap_explanation(
                        explainer, features, feature_names
                    )
                    
                    if fig_waterfall is not None and fig_bar is not None:
                        # Display SHAP plots
                        st.subheader("📈 SHAP Waterfall Plot:")
                        st.pyplot(fig_waterfall)
                        plt.close(fig_waterfall)
                        
                        st.subheader("📊 SHAP Feature Importance:")
                        st.pyplot(fig_bar)
                        plt.close(fig_bar)
                        
                        # Create SHAP values table
                        if shap_values is not None:
                            st.subheader("📋 SHAP Values Table:")
                            shap_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Input Value': [feature_values[name] for name in feature_names],
                                'SHAP Value': shap_values[0],
                                'Contribution': ['Increases Risk' if val > 0 else 'Decreases Risk' for val in shap_values[0]]
                            })
                            
                            # Sort by absolute SHAP value
                            shap_df['Abs_SHAP'] = np.abs(shap_df['SHAP Value'])
                            shap_df = shap_df.sort_values('Abs_SHAP', ascending=False).drop('Abs_SHAP', axis=1)
                            
                            st.dataframe(shap_df.style.format({'SHAP Value': '{:.4f}'}), use_container_width=True)
                            
                            # SHAP interpretation
                            st.info("""
                            **SHAP Values Interpretation:**
                            - **Positive SHAP values** increase the predicted AKI probability
                            - **Negative SHAP values** decrease the predicted AKI probability
                            - **Larger absolute values** indicate greater feature importance
                            - The waterfall plot shows how each feature contributes to the final prediction
                            """)
                    else:
                        st.warning("⚠️ SHAP visualization could not be generated, but prediction is still valid.")
                else:
                    st.warning("⚠️ SHAP explainer could not be created. Prediction results are still valid.")
        else:
            st.info("💡 SHAP analysis is disabled. Enable it above to see feature importance explanations.")
        
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
