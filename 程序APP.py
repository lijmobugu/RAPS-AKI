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

# SHAP 解释器函数
@st.cache_resource
def create_shap_explainer(model):
    """为 StackingClassifier 创建合适的 SHAP 解释器"""
    try:
        # 创建背景数据（这里使用零向量作为简单背景）
        background = np.zeros((1, len(feature_names)))
        
        # 定义预测函数
        def model_predict(X):
            return model.predict_proba(X)
        
        # 使用 KernelExplainer 支持 StackingClassifier
        explainer = shap.KernelExplainer(model_predict, background)
        return explainer
    except Exception as e:
        st.error(f"SHAP 解释器创建失败: {e}")
        return None

# 创建 SHAP 可视化
def create_shap_visualization(explainer, features, feature_names):
    """创建 SHAP 可视化"""
    try:
        if explainer is None:
            return None, None
        
        # 计算 SHAP 值
        shap_values = explainer.shap_values(features.values)
        
        # 如果返回的是列表（多分类），取第一个类别
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # 通常取正类
        
        # 创建 SHAP 图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 创建条形图显示特征重要性
        feature_importance = np.abs(shap_values[0])
        sorted_idx = np.argsort(feature_importance)[::-1]
        
        ax.barh(range(len(feature_names)), feature_importance[sorted_idx])
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.set_xlabel('SHAP Value (Feature Importance)')
        ax.set_title('Feature Importance for Current Prediction')
        
        plt.tight_layout()
        return fig, shap_values
        
    except Exception as e:
        st.error(f"SHAP 可视化创建失败: {e}")
        return None, None

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

# 预测与 SHAP 可视化
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

        # SHAP 解释
        st.subheader("🔍 特征重要性分析")
        
        with st.spinner("正在计算特征重要性..."):
            explainer = create_shap_explainer(model)
            if explainer is not None:
                fig, shap_values = create_shap_visualization(explainer, features, feature_names)
                
                if fig is not None:
                    st.pyplot(fig)
                    plt.close()
                    
                    # 显示数值表格
                    if shap_values is not None:
                        st.subheader("📋 特征贡献度详情")
                        
                        # 创建特征重要性表格
                        importance_df = pd.DataFrame({
                            '特征名称': feature_names,
                            '输入值': [feature_values[name] for name in feature_names],
                            'SHAP值': shap_values[0],
                            '重要性': np.abs(shap_values[0])
                        })
                        
                        # 按重要性排序
                        importance_df = importance_df.sort_values('重要性', ascending=False)
                        
                        # 格式化显示
                        st.dataframe(
                            importance_df.style.format({
                                'SHAP值': '{:.4f}',
                                '重要性': '{:.4f}'
                            }),
                            use_container_width=True
                        )
                        
                        # 解释说明
                        st.info("""
                        **SHAP 值解释:**
                        - 正值表示该特征增加了 AKI 风险
                        - 负值表示该特征降低了 AKI 风险  
                        - 绝对值越大表示该特征对预测结果的影响越大
                        """)
                else:
                    st.warning("⚠️ 特征重要性分析暂时无法显示，但预测结果仍然有效")
            else:
                st.warning("⚠️ 无法创建特征重要性分析，但预测结果仍然有效")

    except Exception as e:
        st.error(f"❌ 预测过程中发生错误: {e}")
        st.info("请检查输入值是否正确，或联系管理员")

# 添加页脚信息
st.markdown("---")
st.markdown("*本预测模型仅供医学研究参考，不能替代专业医疗诊断*")
