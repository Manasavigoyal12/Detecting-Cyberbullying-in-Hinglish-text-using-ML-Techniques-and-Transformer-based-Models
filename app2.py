import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Cyberbullying Detection", layout="wide")

# ---------------- TITLE ----------------
st.title("Hinglish Cyberbullying Detection System")
st.write("Welcome! This system detects cyberbullying in Hinglish text using the MuRIL transformer model.")

st.divider()

# ---------------- LOAD MODEL ----------------
tokenizer = AutoTokenizer.from_pretrained("muril_model")
model = AutoModelForSequenceClassification.from_pretrained("muril_model")
model.eval()

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([2,1])

# ================= LEFT SIDE =================
with col1:

    st.subheader("Enter Hinglish Text")

    text = st.text_area("", placeholder="Example: Tum acche ho")

    if st.button("Analyze Text"):

        if text.strip() == "":
            st.warning("Please enter some text.")
        else:

            text = text.lower()

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            prediction = torch.argmax(logits).item()

            st.divider()

            if prediction == 0:
                st.error("Cyberbullying Detected in this Text")
            else:
                st.success("Non-Cyberbullying!!!")

# ================= RIGHT SIDE =================
with col2:

    st.subheader("Model Analysis")

    # ----------- CONFUSION MATRIX DROPDOWN -----------
    st.markdown("### Confusion Matrix")

    cm_choice = st.selectbox(
        "Select Model for Confusion Matrix",
        ["Select Model", "SVM", "MNB", "MuRIL"]
    )

    if cm_choice == "SVM":
        st.image("images/svm_cm.png")

    elif cm_choice == "MNB":
        st.image("images/mnb_cm.png")

    elif cm_choice == "MuRIL":
        st.image("images/muril_cm.png")

    # ----------- MODEL PERFORMANCE DROPDOWN -----------
    st.markdown("### Model Performance")

    perf_choice = st.selectbox(
        "Select Model for Performance Graph",
        ["Select Model", "SVM", "MNB", "MuRIL"]
    )

    if perf_choice == "SVM":
        st.image("graphs/svm_graph.png")

    elif perf_choice == "MNB":
        st.image("graphs/mnb_graph.png")

    elif perf_choice == "MuRIL":
        st.image("graphs/muril_graph.png")

st.divider()
st.caption("Cyberbullying Detection using MuRIL Transformer")
