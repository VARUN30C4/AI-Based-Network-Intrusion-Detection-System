import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI-Based Network Intrusion Detection System",
    layout="wide"
)

# ------------------ TITLE ------------------
st.title("AI-Based Network Intrusion Detection System")
st.markdown("""
This project uses **Machine Learning (Random Forest Algorithm)** to detect  
**Benign** and **Malicious** network traffic using an interactive dashboard.
""")

# ------------------ DATA GENERATION ------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 5000

    data = {
        "Destination_Port": np.random.randint(1, 65535, n_samples),
        "Flow_Duration": np.random.randint(10, 100000, n_samples),
        "Total_Fwd_Packets": np.random.randint(1, 200, n_samples),
        "Packet_Length_Mean": np.random.uniform(20, 1500, n_samples),
        "Active_Mean": np.random.uniform(1, 1000, n_samples),
        "Label": np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }

    df = pd.DataFrame(data)

    # Simulate attack behavior
    df.loc[df["Label"] == 1, "Total_Fwd_Packets"] += np.random.randint(
        50, 300, size=df[df["Label"] == 1].shape[0]
    )
    df.loc[df["Label"] == 1, "Flow_Duration"] = np.random.randint(
        1, 2000, size=df[df["Label"] == 1].shape[0]
    )

    return df

df = load_data()

# ------------------ SIDEBAR ------------------
st.sidebar.header("Model Configuration")
train_size = st.sidebar.slider("Training Data (%)", 60, 90, 80)
n_estimators = st.sidebar.slider("Number of Trees", 50, 300, 100)

# ------------------ TRAIN TEST SPLIT ------------------
X = df.drop("Label", axis=1)
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=(100 - train_size) / 100, random_state=42
)

# ------------------ MODEL TRAINING ------------------
st.subheader("1️⃣ Model Training")

if st.button("Train Model"):
    with st.spinner("Training the Random Forest model..."):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )
        model.fit(X_train, y_train)
        st.session_state["model"] = model
        st.success("Model trained successfully!")

# ------------------ PERFORMANCE ------------------
st.subheader("2️⃣ Model Performance")

if "model" in st.session_state:
    model = st.session_state["model"]
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{accuracy * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
else:
    st.warning("Please train the model first.")

# ------------------ LIVE TRAFFIC SIMULATOR ------------------
st.subheader("3️⃣ Live Traffic Detection")

col1, col2, col3, col4, col5 = st.columns(5)

dest_port = col1.number_input("Destination Port", 1, 65535, 80)
flow_duration = col2.number_input("Flow Duration", 1, 100000, 500)
total_packets = col3.number_input("Total Forward Packets", 1, 500, 120)
packet_length = col4.number_input("Packet Length Mean", 1, 1500, 600)
active_mean = col5.number_input("Active Mean", 1, 1000, 50)

if st.button("Analyze Traffic"):
    if "model" in st.session_state:
        input_data = np.array([[
            dest_port,
            flow_duration,
            total_packets,
            packet_length,
            active_mean
        ]])

        prediction = st.session_state["model"].predict(input_data)

        if prediction[0] == 1:
            st.error("🚨 MALICIOUS TRAFFIC DETECTED")
        else:
            st.success("✅ BENIGN TRAFFIC (SAFE)")
    else:
        st.error("Please train the model first.")
