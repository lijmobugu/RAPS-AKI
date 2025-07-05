
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

# Create background data for SHAP (smaller dataset for faster computation)
@st.cache_data
def create_background_data():
    """Create background dataset for SHAP explanation"""
    np.random.seed(42)
    n_samples = 20  # Reduced for faster computation
    
    background_data = []
    for _ in range(n_samples):
        sample = {}
        for feature, properties in feature_ranges.items():
            if properties["type"] == "numerical":
                min_val = properties["min"]
                max_val = properties["max"]
                # Use normal distribution around the middle of the range
                mean_val = (min_val + max_val) / 2
                std_val = (max_val - min_val) / 6
                sample[feature] = np.clip(np.random.normal(mean_val, std_val), min_val, max_val)
            elif properties["type"] == "categorical":
                choice = np.random.choice(properties["options"])
                le = LabelEncoder()
                le.fit(properties["options"])
                sample[feature] = le.transform([choice])[0]
        background_data.append(sample)
    
    return pd.DataFrame(background_data, columns=feature_names)

# Simple SHAP analysis for local explanation
def create_local_shap_analysis(model, features, background_data, feature_names):
    """Create local SHAP analysis for the input sample"""
    try:
        # Define prediction function
        def model_predict(X):
            return model.predict_proba(X)[:, 1]  # Return probability of positive class
        
        # Create explainer with smaller background
        explainer = shap.KernelExplainer(model_predict, background_data.values)
        
        # Calculate SHAP values for the input sample (local explanation)
        shap_values = explainer.shap_values(features.values, nsamples=50)
        
        # Get the base value (expected value)
        base_value = explainer.expected_value
        
        # Get the actual prediction
        prediction = model_predict(features.values)[0]
        
        # Extract SHAP values (handle both array and single value cases)
        if isinstance(shap_values, list):
            shap_vals = shap_values[0] if len(shap_values) > 0 else shap_values
        else:
            shap_vals = shap_values[0] if shap_values.ndim > 1 else shap_values
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: SHAP values bar chart
        colors = ['red' if val > 0 else 'blue' for val in shap_vals]
        y_pos = np.arange(len(feature_names))
        
        ax1.barh(y_pos, shap_vals, color=colors, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(feature_names)
        ax1.set_xlabel('SHAP Value (Impact on Prediction)')
        ax1.set_title('Local Feature Importance for This Patient')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(shap_vals):
            ax1.text(v + (0.001 if v >= 0 else -0.001), i, f'{v:.3f}', 
                    va='center', ha='left' if v >= 0 else 'right', fontsize=8)
        
        # Plot 2: Waterfall-style explanation
        # Sort by absolute impact
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
        
        cumulative = base_value
        x_pos = [0]
        y_labels = ['Base\nValue']
        
        for i, idx in enumerate(sorted_idx):
            cumulative += shap_vals[idx]
            x_pos.append(cumulative)
            y_labels.append(f'{feature_names[idx]}\n{shap_vals[idx]:.3f}')
        
        y_labels.append('Final\nPrediction')
        
        # Create waterfall plot
        ax2.plot(range(len(x_pos)), x_pos, 'o-', linewidth=2, markersize=8)
        ax2.set_xticks(range(len(y_labels)))
        ax2.set_xticklabels(y_labels, rotation=45, ha='right')
        ax2.set_ylabel('Prediction Probability')
        ax2.set_title('Prediction Breakdown (Waterfall)')
        ax2.grid(True, alpha=0.3)
        
        # Add horizontal line at base value
        ax2.axhline(y=base_value, color='gray', linestyle='--', alpha=0.5, label=f'Base Value: {base_value:.3f}')
        ax2.axhline(y=prediction, color='red', linestyle='--', alpha=0.5, label=f'Final Prediction: {prediction:.3f}')
        ax2.legend()
        
        plt.tight_layout()
        return fig, shap_vals, base_value, prediction
        
    except Exception as e:
        st.error(f"Local SHAP analysis failed: {e}")
        return None, None, None, None

# Create simple feature importance without SHAP
def create_simple_feature_analysis(model, features, feature_names):
    """Create simple feature importance analysis"""
    try:
        # Get base prediction
        base_pred = model.predict_proba(features)[0, 1]
        
        # Calculate feature importance by perturbation
        feature_importance = []
        
        for i, feature in enumerate(feature_names):
            # Create modified feature set
            modified_features = features.copy()
            
            # For numerical features, try mean value
            if feature_ranges[feature]["type"] == "numerical":
                mean_val = (feature_ranges[feature]["min"] + feature_ranges[feature]["max"]) / 2
                modified_features.iloc[0, i] = mean_val
            else:
                # For categorical, try the opposite value
                current_val = modified_features.iloc[0, i]
                modified_features.iloc[0, i] = 1 - current_val
            
            # Get prediction with modified feature
            modified_pred = model.predict_proba(modified_features)[0, 1]
            
            # Calculate importance as difference
            importance = base_pred - modified_pred
            feature_importance.append(importance)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['red' if val > 0 else 'blue' for val in feature_importance]
        y_pos = np.arange(len(feature_names))
        
        ax.barh(y_pos, feature_importance, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Feature Importance (Perturbation Method)')
        ax.set_title('Feature Importance Analysis for This Patient')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        return fig, feature_importance
        
    except Exception as e:
        st.error(f"Simple feature analysis failed: {e}")
        return None, None

# Streamlit interface
st.title("🏥 AKI Prediction Model with Local SHAP Analysis")
st.header("Please enter the following clinical parameters:")

# Create two-column layout
col1, col2 = st.columns(2)

feature_values = {}
for i, (feature, properties) in enumerate(feature_ranges.items()):
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

# Prediction functionality
if st.button("🔍 Run Local Prediction & Analysis", type="primary"):
    try:
        # Model prediction
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]

        # Display prediction results
        st.subheader("📊 Prediction Results for This Patient:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if predicted_class == 1:
                st.error(f"⚠️ Prediction: High Risk")
                st.error(f"AKI Probability: **{predicted_proba[1]*100:.1f}%**")
            else:
                st.success(f"✅ Prediction: Low Risk")
                st.success(f"AKI Probability: **{predicted_proba[1]*100:.1f}%**")
        
        with col2:
            prob_data = pd.DataFrame({
                'Risk Category': ['Low Risk', 'High Risk'],
                'Probability': [predicted_proba[0]*100, predicted_proba[1]*100]
            })
            st.bar_chart(prob_data.set_index('Risk Category'))
        
        # Local SHAP Analysis
        st.subheader("🔍 Local Feature Importance Analysis:")
        st.info("This analysis explains why the model made this specific prediction for this patient.")
        
        enable_shap = st.checkbox("Enable Local SHAP Analysis", value=True)
        
        if enable_shap:
            analysis_method = st.radio(
                "Choose analysis method:",
                ["SHAP (More Accurate)", "Perturbation (Faster)"],
                help="SHAP provides more accurate explanations but takes longer to compute"
            )
            
            with st.spinner("Analyzing this patient's features..."):
                if analysis_method == "SHAP (More Accurate)":
                    # Create background data
                    background_data = create_background_data()
                    
                    # Run SHAP analysis
                    fig, shap_vals, base_value, prediction = create_local_shap_analysis(
                        model, features, background_data, feature_names
                    )
                    
                    if fig is not None:
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Create detailed table
                        st.subheader("📋 Detailed Feature Impact:")
                        
                        impact_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Patient Value': [feature_values[name] for name in feature_names],
                            'SHAP Impact': shap_vals,
                            'Impact Direction': ['Increases Risk' if val > 0 else 'Decreases Risk' for val in shap_vals],
                            'Absolute Impact': np.abs(shap_vals)
                        })
                        
                        # Sort by absolute impact
                        impact_df = impact_df.sort_values('Absolute Impact', ascending=False)
                        
                        # Format display
                        st.dataframe(
                            impact_df.drop('Absolute Impact', axis=1).style.format({
                                'SHAP Impact': '{:.4f}'
                            }),
                            use_container_width=True
                        )
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Base Risk", f"{base_value:.3f}")
                        with col2:
                            st.metric("Final Risk", f"{prediction:.3f}")
                        with col3:
                            st.metric("Total Impact", f"{prediction - base_value:.3f}")
                            
                else:
                    # Run perturbation analysis
                    fig, importance = create_simple_feature_analysis(model, features, feature_names)
                    
                    if fig is not None:
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Create detailed table
                        st.subheader("📋 Feature Importance (Perturbation Method):")
                        
                        impact_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Patient Value': [feature_values[name] for name in feature_names],
                            'Importance Score': importance,
                            'Impact Direction': ['Increases Risk' if val > 0 else 'Decreases Risk' for val in importance]
                        })
                        
                        # Sort by absolute importance
                        impact_df['Abs_Importance'] = np.abs(impact_df['Importance Score'])
                        impact_df = impact_df.sort_values('Abs_Importance', ascending=False).drop('Abs_Importance', axis=1)
                        
                        st.dataframe(
                            impact_df.style.format({
                                'Importance Score': '{:.4f}'
                            }),
                            use_container_width=True
                        )
            
            # Interpretation
            st.subheader("💡 How to Interpret This Analysis:")
            st.info("""
            **Understanding the Results:**
            - **Red bars**: Features that increase AKI risk for this patient
            - **Blue bars**: Features that decrease AKI risk for this patient
            - **Longer bars**: Features with greater impact on the prediction
            - **Waterfall plot**: Shows how each feature contributes to the final prediction
            
            **This is a LOCAL explanation** - it explains why the model made this specific prediction for this particular patient, not how the model works in general.
            """)
        
        # Clinical interpretation
        st.subheader("🔍 Clinical Interpretation:")
        
        if predicted_class == 1:
            st.warning("""
            **High Risk Prediction for This Patient:**
            - The model predicts a high probability of AKI development
            - Review the feature importance analysis above to understand key risk factors
            - Consider enhanced monitoring and preventive interventions
            - Integrate with clinical judgment and additional risk factors
            """)
        else:
            st.info("""
            **Low Risk Prediction for This Patient:**
            - The model predicts a low probability of AKI development
            - The feature analysis shows which factors are protective
            - Standard monitoring protocols should be maintained
            - Continue to monitor for changes in risk factors
            """)
        
        # Input summary
        st.subheader("📝 Patient Parameters Summary:")
        
        feature_df = pd.DataFrame({
            'Clinical Parameter': feature_names,
            'Input Value': [feature_values[name] for name in feature_names],
            'Data Type': [feature_ranges[name]['type'] for name in feature_names]
        })
        
        st.dataframe(feature_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {e}")
        st.info("Please verify input values are correct or contact the system administrator.")

# Add footer
st.markdown("---")
st.markdown("*This prediction model provides local explanations for individual patients and is for research purposes only*")
