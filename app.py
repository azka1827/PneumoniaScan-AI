import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Set up the website
st.set_page_config(page_title="Pneumonia Detection", page_icon="🩻")

st.title("Pneumonia Detection")
st.write("Upload a chest X-ray to see the model's prediction.")

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)

model.load_state_dict(
    torch.load("best_pneumonia_resnet18.pth", map_location=device)
)

model = model.to(device)
model.eval()

# Prepare the uploaded image
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Upload section
uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")

    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    if st.button("Analyse X-ray"):
        input_image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_image)
            prediction = (output > 0.08).item()

        st.subheader("Model Prediction")

        if prediction == 1:
            st.error("Pneumonia detected")
        else:
            st.success("Normal")

# Disclaimer
st.caption(
    "This project is a prototype and is not medical advice."
)
