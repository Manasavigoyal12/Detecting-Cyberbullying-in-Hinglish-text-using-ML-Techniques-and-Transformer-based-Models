import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------- Page Configuration -----------
st.set_page_config(
    page_title="Cyberbullying Detection",
    page_icon="🤖",
    layout="centered"
)

# ----------- Custom Dark CSS Styling -----------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #00C9A7;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #BBBBBB;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
.cyber {
    background-color: #3a0d0d;
    color: #ff4b4b;
}
.safe {
    background-color: #0d3a1f;
    color: #00ff9d;
}
.footer {
    text-align:center;
    margin-top:40px;
    color:#888888;
}
</style>
""", unsafe_allow_html=True)

# ----------- Title Section -----------
st.markdown('<p class="main-title">🛡 Hinglish Cyberbullying Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Welcome! This system uses a MuRIL Transformer model to detect cyberbullying in Hinglish text.</p>', unsafe_allow_html=True)

st.divider()

# ----------- Load Model -----------
tokenizer = AutoTokenizer.from_pretrained("muril_model")
model = AutoModelForSequenceClassification.from_pretrained("muril_model")
model.eval()

# ----------- Text Input Section -----------
st.markdown("### ✍ Enter Hinglish Text")

text = st.text_area(
    "",
    placeholder="Your Text",
    height=120
)

# ----------- Prediction Button -----------
if st.button("🔍 Analyze Text", use_container_width=True):

    if text.strip() == "":
        st.warning("Please enter some text to analyze.")
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

        probabilities = F.softmax(logits, dim=1)
        confidence = probabilities[0][prediction].item()

        st.divider()

        # ----------- Result Display -----------

        if prediction == 0:
            st.markdown(
                f'<div class="result-box cyber">⚠ Cyberbullying Detected</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box safe">✔ Non-Cyberbullying</div>',
                unsafe_allow_html=True
            )

        # ----------- Confidence Score -----------
        st.write("### Confidence Level")
        st.progress(confidence)

        st.write(f"Confidence Score: **{round(confidence*100,2)} %**")

# ----------- Footer -----------
st.markdown(
    '<p class="footer">AI-powered Cyberbullying Detection using MuRIL Transformer</p>',
    unsafe_allow_html=True
)
