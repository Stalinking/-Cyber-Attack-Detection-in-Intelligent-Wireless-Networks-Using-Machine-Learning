
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Explainable AI Wireless IDS",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Explainable AI-Based Cyber Attack Detection")
st.subheader("Intelligent Wireless Network Security")

st.write(
    "Machine learning-based suspicious traffic "
    "detection with Explainable AI using SHAP."
)

os.makedirs("dataset", exist_ok=True)
os.makedirs("results", exist_ok=True)

FEATURES = [
    "Packet_Rate",
    "Packet_Size",
    "Failed_Authentication",
    "Signal_Strength",
    "Retransmission_Rate",
    "Connection_Count"
]

# ==========================================
# 2. GENERATE SYNTHETIC DATASET
# ==========================================

@st.cache_data
def generate_dataset():

    np.random.seed(42)

    n = 2000
    half = n // 2

    normal = np.column_stack([
        np.random.normal(100, 20, half),
        np.random.normal(500, 80, half),
        np.random.poisson(1, half),
        np.random.normal(-55, 8, half),
        np.random.normal(5, 2, half),
        np.random.normal(10, 3, half)
    ])

    suspicious = np.column_stack([
        np.random.normal(250, 50, half),
        np.random.normal(700, 120, half),
        np.random.poisson(8, half),
        np.random.normal(-70, 10, half),
        np.random.normal(20, 5, half),
        np.random.normal(30, 8, half)
    ])

    data = np.vstack([normal, suspicious])

    labels = np.array(
        [0] * half + [1] * half
    )

    df = pd.DataFrame(
        data,
        columns=FEATURES
    )

    # Keep traffic measurements valid
    for col in FEATURES:
        df[col] = df[col].clip(lower=0)

    df["Label"] = labels

    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    return df


df = generate_dataset()

# Save dataset
df.to_csv(
    "dataset/wireless_network_data.csv",
    index=False
)

st.success("Dataset generated successfully!")

# ==========================================
# 3. DATASET INFORMATION
# ==========================================

st.header("1. Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Normal Records", int((df.Label == 0).sum()))
col3.metric("Suspicious Records", int((df.Label == 1).sum()))

st.dataframe(df.head(10), use_container_width=True)

# ==========================================
# 4. DATA PREPROCESSING
# ==========================================

X = df[FEATURES]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# 5. RANDOM FOREST MODEL
# ==========================================

@st.cache_resource
def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    return model


model = train_model(X_train, y_train)

# ==========================================
# 6. MODEL PREDICTION
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# ==========================================
# 7. PERFORMANCE DASHBOARD
# ==========================================

st.header("2. Machine Learning Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", f"{accuracy * 100:.2f}%")
c2.metric("Precision", f"{precision * 100:.2f}%")
c3.metric("Recall", f"{recall * 100:.2f}%")
c4.metric("F1 Score", f"{f1 * 100:.2f}%")

# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

st.header("3. Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

fig, ax = plt.subplots(figsize=(6, 4))

ax.imshow(cm, cmap="Blues")

ax.set_title("Wireless Traffic Classification")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["Normal", "Suspicious"])
ax.set_yticklabels(["Normal", "Suspicious"])

for i in range(2):
    for j in range(2):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center",
            color="black",
            fontsize=14
        )

plt.tight_layout()

st.pyplot(fig)

fig.savefig(
    "results/confusion_matrix.png",
    dpi=150
)

plt.close(fig)

# ==========================================
# 9. FEATURE IMPORTANCE
# ==========================================

st.header("4. Important Network Features")

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values()

fig, ax = plt.subplots(figsize=(8, 4))

importance.plot(
    kind="barh",
    color="teal",
    ax=ax
)

ax.set_title("Random Forest Feature Importance")
ax.set_xlabel("Importance")

plt.tight_layout()

st.pyplot(fig)

fig.savefig(
    "results/feature_importance.png",
    dpi=150
)

plt.close(fig)

# ==========================================
# 10. SHAP EXPLAINABLE AI
# ==========================================

st.header("5. Explainable AI - SHAP")

@st.cache_resource
def create_explainer(_model):
    return shap.TreeExplainer(_model)


explainer = create_explainer(model)

explanation_data = X_test.iloc[:100]

shap_result = explainer(explanation_data)

# Select class 1: Suspicious traffic
if len(shap_result.values.shape) == 3:

    shap_values = shap_result.values[:, :, 1]

    base_values = shap_result.base_values[:, 1]

else:

    shap_values = shap_result.values

    base_values = shap_result.base_values

# Global SHAP explanation
global_exp = shap.Explanation(
    values=shap_values,
    base_values=base_values,
    data=explanation_data.values,
    feature_names=FEATURES
)

st.write(
    "The SHAP graph explains which features "
    "influence suspicious traffic predictions."
)

fig, ax = plt.subplots(figsize=(9, 5))

shap.plots.beeswarm(
    global_exp,
    max_display=6,
    show=False
)

plt.tight_layout()

st.pyplot(plt.gcf())

plt.savefig(
    "results/shap_explanation.png",
    dpi=150
)

plt.close()

# ==========================================
# 11. INDIVIDUAL PREDICTION
# ==========================================

st.header("6. Test Network Traffic")

st.write(
    "Enter network traffic measurements "
    "to test the trained model."
)

with st.form("traffic_form"):

    packet_rate = st.number_input(
        "Packet Rate",
        min_value=0.0,
        max_value=1000.0,
        value=100.0
    )

    packet_size = st.number_input(
        "Packet Size",
        min_value=0.0,
        max_value=2000.0,
        value=500.0
    )

    failed_auth = st.number_input(
        "Failed Authentication",
        min_value=0,
        max_value=100,
        value=1
    )

    signal_strength = st.number_input(
        "Signal Strength (dBm)",
        min_value=-120.0,
        max_value=0.0,
        value=-55.0
    )

    retransmission = st.number_input(
        "Retransmission Rate",
        min_value=0.0,
        max_value=100.0,
        value=5.0
    )

    connections = st.number_input(
        "Connection Count",
        min_value=0,
        max_value=500,
        value=10
    )

    submitted = st.form_submit_button(
        "Analyze Network Traffic"
    )

if submitted:

    sample = pd.DataFrame(
        [[
            packet_rate,
            packet_size,
            failed_auth,
            signal_strength,
            retransmission,
            connections
        ]],
        columns=FEATURES
    )

    prediction = model.predict(sample)[0]

    probability = model.predict_proba(sample)[0][1]

    st.subheader("Detection Result")

    if prediction == 1:
        st.error("Suspicious Traffic Detected")
    else:
        st.success("Normal Traffic Detected")

    st.metric(
        "Model Probability of Suspicious Class",
        f"{probability * 100:.2f}%"
    )

    # SHAP explanation for this record
    local_result = explainer(sample)

    if len(local_result.values.shape) == 3:

        local_values = local_result.values[0, :, 1]
        local_base = local_result.base_values[0, 1]

    else:

        local_values = local_result.values[0]
        local_base = local_result.base_values[0]

    local_exp = shap.Explanation(
        values=local_values,
        base_values=local_base,
        data=sample.iloc[0].values,
        feature_names=FEATURES
    )

    st.subheader("Why did the model make this prediction?")

    shap.plots.waterfall(
        local_exp,
        show=False
    )

    st.pyplot(plt.gcf())

    plt.savefig(
        "results/individual_explanation.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "M.Tech ECE Project | Explainable AI-Based "
    "Cyber Attack Detection in Intelligent Wireless Networks"
)

st.warning(
    "Research prototype using synthetic data. "
    "Predictions are not validated against real "
    "wireless traffic and should not be used as "
    "a production security system."
)
