
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

# Enhanced feature range definitions with units and reference values
feature_names = [
    "Age", "Diabetes", "WIT", "SCr", "NLR", "DOS", 
    "EBL > 300 mL", "eGFR", "Tumor Dimension", 
    "ASA score"
]

feature_ranges = {
    "Age": {
        "type": "numerical", 
        "min": 18, 
        "max": 80, 
        "default": 50,
        "unit": "years",
        "reference": "Adult: 18-80 years",
        "description": "Patient's age at time of surgery"
    },
    "Diabetes": {
        "type": "categorical", 
        "options": ["YES", "NO"],
        "unit": "Category",
        "reference": "NO: Normal glucose metabolism",
        "description": "Presence of diabetes mellitus"
    },
    "WIT": {
        "type": "numerical", 
        "min": 0, 
        "max": 10, 
        "default": 1.0,
        "unit": "M=min",
        "reference": "Normal: 0.8-1.2",
        "description": "Aspartate aminotransferase to Alanine aminotransferase ratio"
    },
    "SCr": {
        "type": "numerical", 
        "min": 0, 
        "max": 10, 
        "default": 1.0,
        "unit": "mol/L",
        "reference": "Normal: 0.6-1.2 mg/dL (M), 0.5-1.1 mg/dL (F)",
        "description": "Serum creatinine level"
    },
    "NLR": {
        "type": "numerical", 
        "min": 0.5, 
        "max": 5.0, 
        "default": 1.0,
        "unit": "Ratio",
        "reference": "Normal: 0.8-1.2",
        "description": "International Normalized Ratio"
    },
    "DOS": {
        "type": "numerical", 
        "min": 10, 
        "max": 50, 
        "default": 12,
        "unit": "min",
        "reference": "Normal: 11-13 seconds",
        "description": "Prothrombin Time"
    },
    "EBL > 300 mL": {
        "type": "categorical", 
        "options": ["YES", "NO"],
        "unit": "Category",
        "reference": "NO: EBL ≤ 300 mL (Low risk)",
        "description": "Estimated intraoperative blood loss exceeding 300 mL"
    },
    "eGFR": {
        "type": "numerical", 
        "min": 0, 
        "max": 200, 
        "default": 90,
        "unit": "ml/(min×1.73m2)",
        "reference": "Normal: >90 mL/min/1.73m²",
        "description": "Estimated Glomerular Filtration Rate"
    },
    "Tumor Dimension": {
        "type": "numerical", 
        "min": 0, 
        "max": 200, 
        "default": 30,
        "unit": "mm",
        "reference": "Small: <20mm, Medium: 20-40mm, Large: >40mm",
        "description": "Maximum tumor diameter"
    },
    "ASA score": {
        "type": "categorical", 
        "options": ["YES", "NO"],
        "unit": "Grade",
        "reference": "NO: Uncomplicated surgery",
        "description": "Presence of intraoperative complications"
    }
}

# Create background data for SHAP
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
        def model_predict(X):
            return model.predict_proba(X)[:, 1]
        
        explainer = shap.KernelExplainer(model_predict, background_data.values)
        shap_values = explainer.shap_values(features.values, nsamples=50)
        base_value = explainer.expected_value
        prediction = model_predict(features.values)[0]
        
        if isinstance(shap_values, list):
            shap_vals = shap_values[0] if len(shap_values) > 0 else shap_values
        else:
            shap_vals = shap_values[0] if shap_values.ndim > 1 else shap_values
        
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
        
        for i, v in enumerate(shap_vals):
            ax1.text(v + (0.001 if v >= 0 else -0.001), i, f'{v:.3f}', 
                    va='center', ha='left' if v >= 0 else 'right', fontsize=8)
        
        # Plot 2: Waterfall-style explanation
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
        
        cumulative = base_value
        x_pos = [0]
        y_labels = ['Base\nValue']
        
        for i, idx in enumerate(sorted_idx):
            cumulative += shap_vals[idx]
            x_pos.append(cumulative)
            y_labels.append(f'{feature_names[idx]}\n{shap_vals[idx]:.3f}')
        
        y_labels.append('Final\nPrediction')
        
        ax2.plot(range(len(x_pos)), x_pos, 'o-', linewidth=2, markersize=8)
        ax2.set_xticks(range(len(y_labels)))
        ax2.set_xticklabels(y_labels, rotation=45, ha='right')
        ax2.set_ylabel('Prediction Probability')
        ax2.set_title('Prediction Breakdown (Waterfall)')
        ax2.grid(True, alpha=0.3)
        
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
        base_pred = model.predict_proba(features)[0, 1]
        feature_importance = []
        
        for i, feature in enumerate(feature_names):
            modified_features = features.copy()
            
            if feature_ranges[feature]["type"] == "numerical":
                mean_val = (feature_ranges[feature]["min"] + feature_ranges[feature]["max"]) / 2
                modified_features.iloc[0, i] = mean_val
            else:
                current_val = modified_features.iloc[0, i]
                modified_features.iloc[0, i] = 1 - current_val
            
            modified_pred = model.predict_proba(modified_features)[0, 1]
            importance = base_pred - modified_pred
            feature_importance.append(importance)
        
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
st.title("🏥 RAPS-AKI")
st.header("Please enter the following clinical parameters:")

# Add reference values table
with st.expander("📋 Reference Values & Units Guide", expanded=False):
    st.subheader("Clinical Parameter Reference Guide")
    
    ref_data = []
    for feature, properties in feature_ranges.items():
        ref_data.append({
            'Parameter': feature,
            'Unit': properties['unit'],
            'Reference Range': properties['reference'],
            'Description': properties['description']
        })
    
    ref_df = pd.DataFrame(ref_data)
    st.dataframe(ref_df, use_container_width=True)

# Create two-column layout
col1, col2 = st.columns(2)

feature_values = {}
for i, (feature, properties) in enumerate(feature_ranges.items()):
    current_col = col1 if i % 2 == 0 else col2
    
    with current_col:
        if properties["type"] == "numerical":
            # Create enhanced label with unit
            label = f"{feature} ({properties['unit']})"
            
            # Create detailed help text
            help_text = f"""
            **Unit**: {properties['unit']}
            **Reference**: {properties['reference']}
            **Range**: {properties['min']} - {properties['max']}
            **Description**: {properties['description']}
            """
            
            feature_values[feature] = st.number_input(
                label=label,
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
                help=help_text
            )
        elif properties["type"] == "categorical":
            # Create enhanced label
            label = f"{feature}"
            
            # Create detailed help text
            help_text = f"""
            **Reference**: {properties['reference']}
            **Description**: {properties['description']}
            """
            
            feature_values[feature] = st.selectbox(
                label=label,
                options=properties["options"],
                help=help_text
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
        
        # Patient Parameter Summary with Reference Values
        st.subheader("📝 Patient Parameters vs Reference Values:")
        
        # Create comparison table
        comparison_data = []
        for feature in feature_names:
            patient_value = feature_values[feature]
            properties = feature_ranges[feature]
            
            # Determine status
            if properties["type"] == "numerical":
                # Parse reference range for numerical values
                ref_text = properties["reference"]
                if "Normal:" in ref_text:
                    ref_part = ref_text.split("Normal:")[1].strip()
                    status = "Within Normal Range"  # Simplified status
                else:
                    status = "Check Reference"
            else:
                # For categorical variables
                if patient_value == "NO":
                    status = "Normal"
                else:
                    status = "Present"
            
            comparison_data.append({
                'Parameter': feature,
                'Patient Value': f"{patient_value} {properties['unit']}",
                'Reference Range': properties['reference'],
                'Status': status
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
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
                    background_data = create_background_data()
                    fig, shap_vals, base_value, prediction = create_local_shap_analysis(
                        model, features, background_data, feature_names
                    )
                    
                    if fig is not None:
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Create detailed table with units
                        st.subheader("📋 Detailed Feature Impact:")
                        
                        impact_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Patient Value': [f"{feature_values[name]} {feature_ranges[name]['unit']}" for name in feature_names],
                            'SHAP Impact': shap_vals,
                            'Impact Direction': ['Increases Risk' if val > 0 else 'Decreases Risk' for val in shap_vals],
                            'Reference Range': [feature_ranges[name]['reference'] for name in feature_names]
                        })
                        
                        impact_df['Absolute Impact'] = np.abs(impact_df['SHAP Impact'])
                        impact_df = impact_df.sort_values('Absolute Impact', ascending=False)
                        impact_df = impact_df.drop('Absolute Impact', axis=1)
                        
                        st.dataframe(
                            impact_df.style.format({
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
                    fig, importance = create_simple_feature_analysis(model, features, feature_names)
                    
                    if fig is not None:
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        st.subheader("📋 Feature Importance (Perturbation Method):")
                        
                        impact_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Patient Value': [f"{feature_values[name]} {feature_ranges[name]['unit']}" for name in feature_names],
                            'Importance Score': importance,
                            'Impact Direction': ['Increases Risk' if val > 0 else 'Decreases Risk' for val in importance],
                            'Reference Range': [feature_ranges[name]['reference'] for name in feature_names]
                        })
                        
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
            - Compare patient values with reference ranges above
            - Integrate with clinical judgment and additional risk factors
            """)
        else:
            st.info("""
            **Low Risk Prediction for This Patient:**
            - The model predicts a low probability of AKI development
            - The feature analysis shows which factors are protective
            - Patient values comparison with reference ranges shown above
            - Standard monitoring protocols should be maintained
            - Continue to monitor for changes in risk factors
            """)

    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {e}")
        st.info("Please verify input values are correct or contact the system administrator.")

# Add footer
st.markdown("---")
st.markdown("*This prediction model provides local explanations for individual patients and is for research purposes only*")



