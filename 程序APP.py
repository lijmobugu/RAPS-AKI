
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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
    "Age", "Diabetes", "AST/ALT(DRR)", "Creatinine (Cr)", "INR", "PT", 
    "Estimated Blood Loss (EBL) > 300 mL", "eGFR", "Tumor Dimension (mm)", 
    "Intraoperative Complications"
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
    "AST/ALT(DRR)": {
        "type": "numerical", 
        "min": 0, 
        "max": 10, 
        "default": 1.0,
        "unit": "Ratio",
        "reference": "Normal: 0.8-1.2",
        "description": "Aspartate aminotransferase to Alanine aminotransferase ratio"
    },
    "Creatinine (Cr)": {
        "type": "numerical", 
        "min": 0, 
        "max": 10, 
        "default": 1.0,
        "unit": "mg/dL",
        "reference": "Normal: 0.6-1.2 mg/dL (M), 0.5-1.1 mg/dL (F)",
        "description": "Serum creatinine level"
    },
    "INR": {
        "type": "numerical", 
        "min": 0.5, 
        "max": 5.0, 
        "default": 1.0,
        "unit": "Ratio",
        "reference": "Normal: 0.8-1.2",
        "description": "International Normalized Ratio"
    },
    "PT": {
        "type": "numerical", 
        "min": 10, 
        "max": 50, 
        "default": 12,
        "unit": "seconds",
        "reference": "Normal: 11-13 seconds",
        "description": "Prothrombin Time"
    },
    "Estimated Blood Loss (EBL) > 300 mL": {
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
        "unit": "mL/min/1.73m²",
        "reference": "Normal: >90 mL/min/1.73m²",
        "description": "Estimated Glomerular Filtration Rate"
    },
    "Tumor Dimension (mm)": {
        "type": "numerical", 
        "min": 0, 
        "max": 200, 
        "default": 30,
        "unit": "mm",
        "reference": "Small: <20mm, Medium: 20-40mm, Large: >40mm",
        "description": "Maximum tumor diameter"
    },
    "Intraoperative Complications": {
        "type": "categorical", 
        "options": ["YES", "NO"],
        "unit": "Category",
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

# Custom SHAP Force Plot function
def plot_custom_shap_force(shap_values, base_value, pred_value, feature_names, feature_values, font_path=None):
    """Create custom SHAP force plot similar to the uploaded image"""
    try:
        # Sort features by absolute SHAP value for better visualization
        indices = np.argsort(np.abs(shap_values))
        shap_sorted = np.array(shap_values)[indices]
        fname_sorted = np.array(feature_names)[indices]
        fvalue_sorted = np.array(feature_values)[indices]
        
        # Set up font properties
        font_prop = fm.FontProperties(fname=font_path) if font_path else None
        
        # Create figure with appropriate size
        fig, ax = plt.subplots(figsize=(14, 3))
        
        # Start from base value
        start = base_value
        
        # Create bars for each feature
        for i, idx in enumerate(range(len(shap_sorted))):
            val = shap_sorted[idx]
            f_name = fname_sorted[idx]
            f_value = fvalue_sorted[idx]
            
            # Choose color based on positive/negative contribution
            color = "#ff4444" if val > 0 else "#4444ff"
            
            # Create horizontal bar
            ax.barh(0, val, left=start, color=color, alpha=0.8, height=0.3, 
                   edgecolor='white', linewidth=1)
            
            # Add text label if the contribution is significant
            if abs(val) > 0.005:  # Only show labels for significant contributions
                label_text = f"{f_name} = {f_value}"
                text_x = start + val / 2
                
                # Choose text color for visibility
                text_color = 'white' if abs(val) > 0.02 else 'black'
                
                ax.text(text_x, 0, label_text, 
                       ha='center', va='center', color=text_color, 
                       fontsize=9, fontweight='bold',
                       fontproperties=font_prop,
                       bbox=dict(boxstyle="round,pad=0.1", facecolor=color, alpha=0.3) if text_color == 'black' else None)
            
            start += val
        
        # Add base value and prediction lines
        ax.axvline(base_value, color='gray', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(pred_value, color='black', linestyle='-', linewidth=2, alpha=0.8)
        
        # Add text annotations for base value and prediction
        ax.text(base_value, 0.4, f'Base Value\n{base_value:.3f}', 
               ha='center', va='bottom', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7),
               fontproperties=font_prop)
        
        ax.text(pred_value, 0.4, f'Prediction\n{pred_value:.3f}', 
               ha='center', va='bottom', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7),
               fontproperties=font_prop)
        
        # Format axes
        ax.set_xlim(min(base_value, pred_value) - 0.1, max(base_value, pred_value) + 0.1)
        ax.set_ylim(-0.3, 0.6)
        ax.set_xlabel('Model Output (AKI Risk Probability)', fontsize=12, fontweight='bold', fontproperties=font_prop)
        ax.set_title(f'SHAP Force Plot - Based on feature values, predicted possibility of AKI is {pred_value*100:.2f}%', 
                    fontsize=14, fontweight='bold', pad=20, fontproperties=font_prop)
        
        # Remove y-axis and ticks
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Add grid for better readability
        ax.grid(True, axis='x', alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        st.error(f"Failed to create SHAP force plot: {e}")
        return None

# Simple SHAP analysis for local explanation
def create_local_shap_analysis(model, features, background_data, feature_names, feature_values):
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
        
        # Create the force plot
        force_fig = plot_custom_shap_force(
            shap_vals, base_value, prediction, 
            feature_names, [feature_values[name] for name in feature_names]
        )
        
        # Create traditional analysis plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: SHAP values bar chart
        colors = ['#ff4444' if val > 0 else '#4444ff' for val in shap_vals]
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
        x_pos = [cumulative]
        y_labels = ['Base Value']
        
        for i, idx in enumerate(sorted_idx):
            cumulative += shap_vals[idx]
            x_pos.append(cumulative)
            y_labels.append(f'{feature_names[idx]}\n{shap_vals[idx]:.3f}')
        
        ax2.plot(range(len(x_pos)), x_pos, 'o-', linewidth=2, markersize=8, color='#2E86AB')
        ax2.set_xticks(range(len(y_labels)))
        ax2.set_xticklabels(y_labels, rotation=45, ha='right')
        ax2.set_ylabel('Prediction Probability')
        ax2.set_title('Prediction Breakdown (Waterfall)')
        ax2.grid(True, alpha=0.3)
        
        ax2.axhline(y=base_value, color='gray', linestyle='--', alpha=0.5, label=f'Base Value: {base_value:.3f}')
        ax2.axhline(y=prediction, color='red', linestyle='--', alpha=0.5, label=f'Final Prediction: {prediction:.3f}')
        ax2.legend()
        
        plt.tight_layout()
        return fig, force_fig, shap_vals, base_value, prediction
        
    except Exception as e:
        st.error(f"Local SHAP analysis failed: {e}")
        return None, None, None, None, None

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
        
        colors = ['#ff4444' if val > 0 else '#4444ff' for val in feature_importance]
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
st.title("🏥 AKI Prediction Model with SHAP Force Plot")
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
if st.button("🔍 Run Prediction & SHAP Analysis", type="primary"):
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
                st.error(f"AKI Probability: **{predicted_proba[1]*100:.2f}%**")
            else:
                st.success(f"✅ Prediction: Low Risk")
                st.success(f"AKI Probability: **{predicted_proba[1]*100:.2f}%**")
        
        with col2:
            prob_data = pd.DataFrame({
                'Risk Category': ['Low Risk', 'High Risk'],
                'Probability': [predicted_proba[0]*100, predicted_proba[1]*100]
            })
            st.bar_chart(prob_data.set_index('Risk Category'))
        
        # SHAP Force Plot
        st.subheader("🎯 SHAP Force Plot - Feature Contribution Analysis")
        
        with st.spinner("Generating SHAP force plot..."):
            background_data = create_background_data()
            fig, force_fig, shap_vals, base_value, prediction = create_local_shap_analysis(
                model, features, background_data, feature_names, feature_values
            )
            
            if force_fig is not None:
                st.pyplot(force_fig)
                plt.close(force_fig)
                
                st.info("""
                **How to read the SHAP Force Plot:**
                - 🔴 **Red sections**: Features that increase AKI risk
                - 🔵 **Blue sections**: Features that decrease AKI risk  
                - The plot shows how each feature pushes the prediction from the base value to the final prediction
                """)
            
            # Traditional SHAP plots
            if fig is not None:
                st.subheader("📊 Detailed SHAP Analysis")
                st.pyplot(fig)
                plt.close(fig)
        
        # Patient Parameter Summary
        st.subheader("📝 Patient Parameters vs Reference Values:")
        
        comparison_data = []
        for feature in feature_names:
            patient_value = feature_values[feature]
            properties = feature_ranges[feature]
            
            # Determine status
            if properties["type"] == "numerical":
                status = "Within Normal Range"
            else:
                status = "Normal" if patient_value == "NO" else "Present"
            
            comparison_data.append({
                'Parameter': feature,
                'Patient Value': f"{patient_value} {properties['unit']}",
                'Reference Range': properties['reference'],
                'Status': status
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Feature Impact Table
        if 'shap_vals' in locals() and shap_vals is not None:
            st.subheader("📋 Feature Impact Summary:")
            
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
        
        # Clinical interpretation
        st.subheader("🔍 Clinical Interpretation:")
        
        if predicted_class == 1:
            st.warning("""
            **High Risk Prediction for This Patient:**
            - The model predicts a high probability of AKI development
            - Review the SHAP force plot to understand key contributing factors
            - Consider enhanced monitoring and preventive interventions
            - Integrate with clinical judgment and additional risk factors
            """)
        else:
            st.info("""
            **Low Risk Prediction for This Patient:**
            - The model predicts a low probability of AKI development
            - The SHAP analysis shows protective factors
            - Standard monitoring protocols should be maintained
            - Continue to monitor for changes in risk factors
            """)

    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {e}")
        st.info("Please verify input values are correct or contact the system administrator.")

# Add footer
st.markdown("---")
st.markdown("*This prediction model provides local explanations for individual patients and is for research purposes only*")
