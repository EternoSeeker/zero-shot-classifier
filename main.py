import streamlit as st
from transformers import pipeline

# Load zero-shot classification pipeline
@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_model()

# Streamlit UI
st.title("🧠 Zero-Shot Text Classifier")
st.write("Classify text into custom labels without training!")

# User input
text_input = st.text_area("Enter your text:")
labels_input = st.text_input("Enter candidate labels (comma-separated):")

if st.button("Classify"):
    if text_input and labels_input:
        labels = [label.strip() for label in labels_input.split(",")]
        with st.spinner("Classifying..."):
            result = classifier(text_input, candidate_labels=labels)
        st.subheader("Prediction")
        st.write(f"**Text:** {text_input}")
        st.write(f"**Labels:** {labels}")
        st.write("**Scores:**")
        for label, score in zip(result["labels"], result["scores"]):
            st.write(f"- {label}: {score:.4f}")
    else:
        st.warning("Please provide both text and candidate labels.")