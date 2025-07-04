import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder  # ← 这行很重要
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import catboost
import shap
# 暂时注释掉这些可能有问题的导入
# from catboost import CatBoostClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
# 加载模型
try:
    model = joblib.load('final_stacking_model.pkl')
except (FileNotFoundError, AttributeError, ModuleNotFoundError) as e:
    st.error(f"模型加载失败：{e}")
    st.info("模型文件与当前环境不兼容，请重新训练模型或检查依赖版本")
    model = None
    st.stop()


# 特征范围定义
feature_names = [
    "Age", "Diabetes", "AST/ALT(DRR)", "Creatinine (Cr)", "INR", "PT", "Estimated Blood Loss (EBL) > 300 mL", "eGFR", "Tumor Dimension (mm)","Intraoperative Complications"]
feature_ranges = {
    "Age": {"type": "numerical", "min": 0, "max": 200, "default": 0},
    "Diabetes": {"type": "categorical", "options": ["YES", "NO"]},
    "AST/ALT(DRR)": {"type": "numerical", "min": 18, "max": 80, "default": 40},
    "Creatinine (Cr)": {"type": "numerical", "min": 0, "max": 170, "default": 0},
    "INR": {"type": "numerical", "min": 140, "max": 170, "default": 160},
    "PT": {"type": "numerical", "min": 18, "max": 80, "default": 40},
    "Estimated Blood Loss (EBL) > 300 mL": {"type": "categorical", "options": ["YES", "NO"]},
    "eGFR": {"type": "numerical", "min": 18, "max": 80, "default": 40},
    "Tumor Dimension (mm)": {"type": "numerical", "min": 0, "max": 170, "default": 0},
    "Intraoperative Complications": {"type": "categorical", "options": ["YES", "NO"]}
}

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")
st.header("Enter the following feature values:")

feature_values = {}
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        feature_values[feature] = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        feature_values[feature] = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )

# 处理分类特征
label_encoders = {}
for feature, properties in feature_ranges.items():
    if properties["type"] == "categorical":
        label_encoders[feature] = LabelEncoder()
        label_encoders[feature].fit(properties["options"])
        feature_values[feature] = label_encoders[feature].transform([feature_values[feature]])[0]

# 转换为模型输入格式
features = pd.DataFrame([feature_values], columns=feature_names)
