import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ========== 字体配置 ==========
FONT_PATH = '/mnt/data/file-ngwyeoEN29l1M3O1QpdxCwkj-sider-font.ttf'
font_prop = fm.FontProperties(fname=FONT_PATH)
plt.rcParams['axes.unicode_minus'] = False

# ========== 基础配置 ==========
feature_names = [
    "Age", "Diabetes", "AST/ALT(DRR)", "Creatinine (Cr)", "INR", "PT", 
    "Estimated Blood Loss (EBL) > 300 mL", "eGFR", "Tumor Dimension (mm)", 
    "Intraoperative Complications"
]
feature_ranges = {
    "Age": {"type": "numerical", "min": 18, "max": 80, "default": 50, "info": "年龄（岁）"},
    "Diabetes": {"type": "categorical", "options": ["YES", "NO"], "info": "糖尿病史"},
    "AST/ALT(DRR)": {"type": "numerical", "min": 0, "max": 10, "default": 1.0, "info": "AST/ALT 比值"},
    "Creatinine (Cr)": {"type": "numerical", "min": 0, "max": 10, "default": 1.0, "info": "肌酐"},
    "INR": {"type": "numerical", "min": 0.5, "max": 5.0, "default": 1.0, "info": "国际标准化比值"},
    "PT": {"type": "numerical", "min": 10, "max": 50, "default": 12, "info": "凝血酶原时间"},
    "Estimated Blood Loss (EBL) > 300 mL": {"type": "categorical", "options": ["YES", "NO"], "info": "术中出血量 >300ml"},
    "eGFR": {"type": "numerical", "min": 0, "max": 200, "default": 90, "info": "肾小球滤过率"},
    "Tumor Dimension (mm)": {"type": "numerical", "min": 0, "max": 200, "default": 30, "info": "肿瘤最大径(mm)"},
    "Intraoperative Complications": {"type": "categorical", "options": ["YES", "NO"], "info": "术中并发症"}
}

# ========== 模型加载 ==========
@st.cache_resource
def load_model():
    try:
        model = joblib.load('final_stacking_model.pkl')
        return model
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        st.stop()
model = load_model()

# ========== 输入界面 ==========
st.title("🏥 AKI术后肾损伤智能预测")
st.header("请输入患者关键临床特征：")

cols = st.columns(2)
feature_values = {}
for idx, (f, props) in enumerate(feature_ranges.items()):
    c = cols[idx % 2]
    with c:
        if props['type'] == 'numerical':
            feature_values[f] = st.number_input(
                f, float(props['min']), float(props['max']), float(props['default']),
                help=props.get('info','')
            )
        else:
            feature_values[f] = st.selectbox(
                f, props['options'], help=props.get('info','')
            )

# ========== 预处理 ==========
def process_input(features:dict, feature_ranges:dict):
    out = features.copy()
    for f, props in feature_ranges.items():
        if props['type'] == "categorical":
            le = LabelEncoder()
            le.fit(props["options"])
            out[f] = le.transform([features[f]])[0]
    return pd.DataFrame([out], columns=feature_names)
features = process_input(feature_values, feature_ranges)

# ========== 绘制概率图表（matplotlib风格） ==========
def plot_prob_bar(proba, font_prop):
    fig, ax = plt.subplots(figsize=(3,2))
    ax.bar(['低风险','高风险'], [proba[0]*100, proba[1]*100], color=['#1766ad', '#b70404'])
    ax.set_ylabel("概率 (%)", fontproperties=font_prop)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontproperties(font_prop)
    st.pyplot(fig)
    plt.close(fig)

# ========== 预测&结果展示 ==========
if st.button("🔍 运行预测"):
    try:
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        col1, col2 = st.columns(2)
        with col1:
            if pred == 1:
                st.error(f"⚠️ 高风险\nAKI概率: **{proba[1]*100:.2f}%**")
            else:
                st.success(f"✅ 低风险\nAKI概率: **{proba[1]*100:.2f}%**")
        with col2:
            plot_prob_bar(proba, font_prop)
        # 明细表
        prob_df = pd.DataFrame({
            '风险类别':['低风险(类0)','高风险(类1)'],
            '预测概率':[f"{proba[0]*100:.2f}%", f"{proba[1]*100:.2f}%"],
            '置信度':[f"{proba[0]:.4f}", f"{proba[1]:.4f}"]
        })
        st.dataframe(prob_df, use_container_width=True)
        # 输入参数回顾
        st.subheader("输入参数回顾：")
        feat_df = pd.DataFrame({
            '参数': list(feature_values.keys()),
            '输入值': list(feature_values.values()),
            '类型': [feature_ranges[k]['type'] for k in feature_values]
        })
        st.dataframe(feat_df, use_container_width=True)
        # 诊断解释
        st.subheader("🤔 诊断建议")
        if pred == 1:
            st.warning("模型预测高风险，建议加强术后监测和干预")
        else:
            st.info("模型预测低风险，请常规随访。")
    except Exception as e:
        st.error(f"❌ 预测失败: {e}")

# ----- 尾注 --------
st.markdown("---")
st.markdown("*本医学预测工具仅供学术参考，不能替代专业医生判断*")
