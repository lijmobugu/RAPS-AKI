
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

# Streamlit 界面
st.title("🏥 AKI 预测模型")
st.header("请输入以下特征值:")

# 创建两列布局
col1, col2 = st.columns(2)

feature_values = {}
for i, (feature, properties) in enumerate(feature_ranges.items()):
    # 交替放置在两列中
    current_col = col1 if i % 2 == 0 else col2
    
    with current_col:
        if properties["type"] == "numerical":
            feature_values[feature] = st.number_input(
                label=f"{feature}",
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
                help=f"范围: {properties['min']} - {properties['max']}"
            )
        elif properties["type"] == "categorical":
            feature_values[feature] = st.selectbox(
                label=f"{feature}",
                options=properties["options"],
            )

# 处理分类特征
processed_values = feature_values.copy()
label_encoders = {}

for feature, properties in feature_ranges.items():
    if properties["type"] == "categorical":
        label_encoders[feature] = LabelEncoder()
        label_encoders[feature].fit(properties["options"])
        processed_values[feature] = label_encoders[feature].transform([feature_values[feature]])[0]

# 转换为模型输入格式
features = pd.DataFrame([processed_values], columns=feature_names)

# 预测功能（无 SHAP）
if st.button("🔍 开始预测", type="primary"):
    try:
        # 模型预测
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]

        # 显示预测结果
        st.subheader("📊 预测结果:")
        
        # 创建结果展示
        col1, col2 = st.columns(2)
        
        with col1:
            if predicted_class == 1:
                st.error(f"⚠️ 预测结果: 高风险")
                st.error(f"AKI 发生概率: **{predicted_proba[1]*100:.1f}%**")
            else:
                st.success(f"✅ 预测结果: 低风险")
                st.success(f"AKI 发生概率: **{predicted_proba[1]*100:.1f}%**")
        
        with col2:
            # 显示概率分布
            prob_data = pd.DataFrame({
                '类别': ['低风险', '高风险'],
                '概率': [predicted_proba[0]*100, predicted_proba[1]*100]
            })
            st.bar_chart(prob_data.set_index('类别'))
        
        # 详细概率信息
        st.subheader("📋 详细预测信息:")
        
        # 创建概率表格
        prob_df = pd.DataFrame({
            '风险类别': ['低风险 (Class 0)', '高风险 (Class 1)'],
            '预测概率': [f"{predicted_proba[0]*100:.2f}%", f"{predicted_proba[1]*100:.2f}%"],
            '置信度': [f"{predicted_proba[0]:.4f}", f"{predicted_proba[1]:.4f}"]
        })
        
        st.dataframe(prob_df, use_container_width=True)
        
        # 风险解释
        st.subheader("🔍 结果解释:")
        
        if predicted_class == 1:
            st.warning("""
            **高风险预测说明:**
            - 模型预测该患者发生 AKI 的概率较高
            - 建议加强监护和预防措施
            - 请结合临床实际情况进行综合判断
            """)
        else:
            st.info("""
            **低风险预测说明:**
            - 模型预测该患者发生 AKI 的概率较低
            - 仍需要常规监护
            - 请结合临床实际情况进行综合判断
            """)
        
        # 输入特征回顾
        st.subheader("📝 输入特征回顾:")
        
        # 创建特征表格
        feature_df = pd.DataFrame({
            '特征名称': feature_names,
            '输入值': [feature_values[name] for name in feature_names],
            '数据类型': [feature_ranges[name]['type'] for name in feature_names]
        })
        
        st.dataframe(feature_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 预测过程中发生错误: {e}")
        st.info("请检查输入值是否正确，或联系管理员")

# 添加页脚信息
st.markdown("---")
st.markdown("*本预测模型仅供医学研究参考，不能替代专业医疗诊断*")
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


st.error(f"An error occurred: {e}")
