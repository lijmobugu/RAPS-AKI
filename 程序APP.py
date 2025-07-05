import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import catboost

# 字体设置
FONT_PATH = '/mnt/data/file-ngwyeoEN29l1M3O1QpdxCwkj-sider-font.ttf'
font_prop = fm.FontProperties(fname=FONT_PATH)
plt.rcParams['axes.unicode_minus'] = False

# ------- 功能函数定义 -------

def load_model_safely(model_path='final_stacking_model.pkl'):
    try:
        model = joblib.load(model_path)
        return model
    except (FileNotFoundError, AttributeError, ModuleNotFoundError) as e:
        st.error(f"Model loading failed: {e}")
        st.info("Model file is incompatible with current environment. Please retrain the model or check dependency versions.")
        st.stop()

def build_input_form(feature_ranges):
    st.header("请填写临床参数（单例）：")
    col1, col2 = st.columns(2)
    input_data = {}
    i = 0
    for feature, props in feature_ranges.items():
        cc = col1 if i % 2 == 0 else col2
        with cc:
            if props['type'] == 'numerical':
                input_data[feature] = st.number_input(
                    label=f"{feature}",
                    min_value=float(props["min"]),
                    max_value=float(props["max"]),
                    value=float(props["default"]),
                    help=f"范围: {props['min']} - {props['max']}"
                )
            else:
                input_data[feature] = st.selectbox(
                    label=f"{feature}",
                    options=props["options"],
                )
        i += 1
    return input_data

def encode_categoricals(input_data, feature_ranges):
    processed = input_data.copy()
    label_encoders = {}
    for feature, props in feature_ranges.items():
        if props["type"] == "categorical":
            le = LabelEncoder()
            le.fit(props["options"])
            processed[feature] = le.transform([input_data[feature]])[0]
            label_encoders[feature] = le
    return processed, label_encoders

def get_feature_importance(model, feature_names):
    # 适配不同sklearn集成/boosting模型
    try:
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        elif hasattr(model, 'named_estimators_'):
            # stacking模型
            for est in model.named_estimators_.values():
                if hasattr(est, "feature_importances_"):
                    return est.feature_importances_
        return None
    except:
        return None

def plot_probability_pie(probs, categories, font_prop):
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        probs,
        labels=categories,
        autopct='%1.1f%%',
        textprops=dict(fontproperties=font_prop)
    )
    for text in texts + autotexts:
        text.set_fontproperties(font_prop)
    ax.set_title("AKI 风险概率分布", fontproperties=font_prop)
    st.pyplot(fig)
    plt.close(fig)

def plot_feature_importance_bar(importances, feature_names, font_prop):
    fig, ax = plt.subplots(figsize=(6, 4))
    sort_idx = np.argsort(importances)[::-1]
    names = np.array(feature_names)[sort_idx]
    vals = np.array(importances)[sort_idx]
    ax.barh(names, vals)
    ax.invert_yaxis()
    ax.set_xlabel("特征重要性", fontproperties=font_prop)
    ax.set_title("模型特征重要性", fontproperties=font_prop)
    ax.tick_params(axis='y', labelsize=9)
    for label in ax.get_yticklabels(): label.set_fontproperties(font_prop)
    for label in ax.get_xticklabels(): label.set_fontproperties(font_prop)
    st.pyplot(fig)
    plt.close(fig)

def preprocess_batch_data(df, feature_ranges):
    encoders = {}
    for feature, props in feature_ranges.items():
        if props['type'] == 'categorical':
            le = LabelEncoder()
            le.fit(props["options"])
            df[feature] = le.transform(df[feature].astype(str))
            encoders[feature] = le
    return df

def download_dataframe(df, filename):
    csv = df.to_csv(index=False).encode()
    st.download_button(
        label="下载预测结果",
        data=csv,
        file_name=filename,
        mime='text/csv',
    )

# ------- 配置部分 -------

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

# ------- 主页面 -------

st.set_page_config(page_title="AKI智能预测模型", layout="wide")
st.title("🏥 AKI 术后肾损伤风险预测模型")

choice = st.sidebar.radio("功能选择", ["单例预测", "批量预测"])

model = load_model_safely()

# ------- 单例预测 -------

if choice == "单例预测":
    input_data = build_input_form(feature_ranges)
    processed, label_encoders = encode_categoricals(input_data, feature_ranges)
    input_df = pd.DataFrame([processed], columns=feature_names)
    
    if st.button("🔍 运行预测", type='primary'):
        try:
            with st.spinner("模型正在预测..."):
                pred_class = model.predict(input_df)[0]
                pred_proba = model.predict_proba(input_df)[0]

                # 结果展示
                st.subheader("📊 预测结果")
                col1, col2 = st.columns(2)
                with col1:
                    if pred_class == 1:
                        st.error(f"⚠️ 高风险")
                        st.error(f"AKI 概率: **{pred_proba[1]*100:.2f}%**")
                    else:
                        st.success(f"✅ 低风险")
                        st.success(f"AKI 概率: **{pred_proba[1]*100:.2f}%**")
                with col2:
                    plot_probability_pie(
                        [pred_proba[0]*100, pred_proba[1]*100],
                        ["低风险", "高风险"],
                        font_prop
                    )
                # 概率表
                prob_df = pd.DataFrame({
                    '风险类别': ['低风险(类0)', '高风险(类1)'],
                    '预测概率': [f"{pred_proba[0]*100:.2f}%", f"{pred_proba[1]*100:.2f}%"],
                    '置信度': [f"{pred_proba[0]:.4f}", f"{pred_proba[1]:.4f}"]
                })
                st.dataframe(prob_df, use_container_width=True)
                # 风险解释
                st.subheader("🔍 临床解释")
                if pred_class == 1:
                    st.warning("**高风险**: 建议加强监测和预防干预（具体措施请结合实际临床环境）")
                else:
                    st.info("**低风险**: 标准术后监测即可。")
                # 参数汇总
                st.subheader("📝 输入参数汇总")
                featdf = pd.DataFrame({
                    '参数': feature_names,
                    '输入值': [input_data[f] for f in feature_names],
                    '类型': [feature_ranges[f]['type'] for f in feature_names]
                })
                st.dataframe(featdf, use_container_width=True)

                # 特征重要性
                st.subheader("📈 特征重要性")
                importances = get_feature_importance(model, feature_names)
                if importances is not None:
                    plot_feature_importance_bar(importances, feature_names, font_prop)
                else:
                    st.info("⚠️ 当前模型不支持特征重要性解释。")

                # 下载单次结果
                single_result_df = input_df.copy()
                single_result_df['AKI风险概率'] = pred_proba[1]
                single_result_df['AKI分类'] = pred_class
                download_dataframe(single_result_df, "single_prediction_result.csv")

        except Exception as e:
            st.error(f"❌ 预测出错: {e}")

# ------- 批量预测 -------

if choice == "批量预测":
    st.header("批量预测（上传CSV，需要表头）")
    st.markdown("应包含列：" + ", ".join([f"`{x}`" for x in feature_names]))
    batch_file = st.file_uploader("上传CSV文件", type=["csv"])
    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            # 自动适配列名顺序
            batch_df = batch_df[[f for f in feature_names]]
            st.success("上传成功，数据如下：")
            st.dataframe(batch_df.head(), use_container_width=True)
            # 数据预处理
            proc_batch_df = preprocess_batch_data(batch_df.copy(), feature_ranges)
            
            if st.button("批量运行预测"):
                with st.spinner("正在批量预测..."):
                    y_pred = model.predict(proc_batch_df)
                    y_proba = model.predict_proba(proc_batch_df)[:, 1]
                    out_df = batch_df.copy()
                    out_df['AKI风险概率'] = y_proba
                    out_df['AKI分类'] = y_pred
                    st.success("预测完成！主要结果：")
                    st.dataframe(out_df.head(), use_container_width=True)
                    download_dataframe(out_df, "batch_prediction_result.csv")
                    # 简单统计图
                    st.subheader("批量预测统计")
                    fig, ax = plt.subplots()
                    labels = ['低风险', '高风险']
                    counts = [(y_pred==0).sum(), (y_pred==1).sum()]
                    ax.bar(labels, counts)
                    ax.set_ylabel("人数", fontproperties=font_prop)
                    ax.set_title("风险分类统计", fontproperties=font_prop)
                    for label in ax.get_xticklabels() + ax.get_yticklabels():
                        label.set_fontproperties(font_prop)
                    st.pyplot(fig)
                    plt.close(fig)

        except Exception as e:
            st.error(f"❌ 批量预测失败: {e}")

# -------- 尾注 --------

st.markdown("---")
st.markdown("*本模型用于医学学术研究辅助预测，不作为直接医疗依据，与实际临床决策配合使用。*")
