import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Pneumonia vs Healthy",
    page_icon="🩻",
    layout="wide"
)

# Website design
st.title("Pneumonia vs Healthy")
st.subheader("AI-powered Chest X-ray Classification")
st.write(
    "Upload a chest X-ray to explore the prediction "
    "made by our AI model."
)

st.divider()

# Load trained model
@st.cache_resource
def load_model():
    device = torch.device("cpu")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)

    model.load_state_dict(
        torch.load(
            "Pneumonia_vs_Healthy.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model, device

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

# User input
st.header("Upload your X-ray")

case_notes = st.text_area(
    "Optional case notes",
    placeholder="Add a case label or notes..."
)

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded chest X-ray",
            use_container_width=True
        )

    with col2:
        st.write("**Case notes**")
        st.write(case_notes if case_notes else "No notes provided.")

    if st.button("Analyse X-ray", type="primary"):

        try:
            model, device = load_model()

            input_image = transform(image).unsqueeze(0).to(device)

            # Generate prediction
            with torch.no_grad():
                output = model(input_image)
                probability = torch.sigmoid(output).item()

            prediction = output.item() > 0.08

            # Display results
            st.divider()
            st.header("AI Analysis Results")

            if prediction:
                st.error("Prediction: Pneumonia")
                confidence = probability * 100
            else:
                st.success("Prediction: Normal")
                confidence = (1 - probability) * 100

            st.metric(
                "Model confidence",
                f"{confidence:.2f}%"
            )

            st.progress(float(confidence / 100))

            st.caption(
                "Confidence is based on the model's output "
                "and is not a measure of diagnostic certainty."
            )

            # Generate Grad-CAM visual explanation
            st.subheader("Model Attention Heatmap")

            activations = []
            gradients = []

            def forward_hook(module, input, output):
                activations.append(output)
                output.register_hook(
                    lambda grad: gradients.append(grad)
                )

            target_layer = model.layer4[-1]
            handle = target_layer.register_forward_hook(forward_hook)

            model.zero_grad()
            grad_output = model(input_image)
            grad_output[0, 0].backward()

            handle.remove()

            activation = activations[0].detach()
            gradient = gradients[0].detach()

            weights = gradient.mean(dim=(2, 3), keepdim=True)

            cam = (weights * activation).sum(dim=1)
            cam = torch.relu(cam)

            cam = cam.squeeze().cpu().numpy()

            if cam.max() > 0:
                cam = cam / cam.max()

            cam_image = Image.fromarray(
                np.uint8(cam * 255)
            ).resize(image.size)

            fig, ax = plt.subplots()
            ax.imshow(image, cmap="gray")
            ax.imshow(
                np.array(cam_image),
                cmap="jet",
                alpha=0.4
            )
            ax.axis("off")

            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "The heatmap highlights image regions that "
                "influenced the model. It does not identify "
                "confirmed symptoms or disease locations."
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")

st.divider()

# Disclaimer
st.warning(
    "This project is a prototype and is not medical advice. "
    "It must not be used to diagnose or make treatment decisions."
)
