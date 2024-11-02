import streamlit as st
from PIL import Image
import numpy as np
import torch
from transformers import ViTForImageClassification, ViTImageProcessor

# Function to load model
def load_model():
    model_name = "google/vit-base-patch16-224-in21k"  # Replace with your chosen model
    model = ViTForImageClassification.from_pretrained(model_name)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    return model, feature_extractor

# Function to predict disease
def predict_disease(image, model, feature_extractor):
    image = image.convert("RGB")
    image_array = np.array(image)
    
    if image_array.ndim == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)
    
    inputs = feature_extractor(images=image_array, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
    return predicted_class_idx

# Function to map labels to disease names
def get_disease_name(label):
    class_names = {
        0: "Normal",
        1: "Pneumonia",
        2: "Tuberculosis",
        3: "Lung Cancer",
        4: "Other Disease"
    }
    return class_names.get(label, "Unknown Disease")

# Streamlit app
st.title("Comprehensive Medical Image Analysis")
st.write("Upload X-ray or CT scan images to get a detailed disease analysis.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    model, feature_extractor = load_model()
    
    if st.button("Predict"):
        predicted_class_idx = predict_disease(image, model, feature_extractor)
        disease_name = get_disease_name(predicted_class_idx)
        st.write(f"Predicted Disease: {disease_name}")

        explanation = {
            "Normal": "No signs of disease detected.",
            "Pneumonia": "Pneumonia is an infection that inflames the air sacs in one or both lungs.",
            "Tuberculosis": "Tuberculosis is a potentially serious infectious disease that mainly affects the lungs.",
            "Lung Cancer": "Lung cancer is a type of cancer that begins in the lungs.",
            "Other Disease": "Further medical evaluation is recommended."
        }
        st.write(explanation.get(disease_name, "No detailed explanation available."))
