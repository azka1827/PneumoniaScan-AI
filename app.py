```python
# app.py
# PneumoniaScan AI
# Experimental chest X-ray classification website

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

from PIL import Image


# -----------------------------
# 1. WEBSITE SETTINGS
# -----------------------------

st.set_page_config(
    page_title="PneumoniaScan AI",
    page_icon="🫁",
    layout="wide"
)

st.title("PneumoniaScan AI")
st.write("An experimental AI tool for analysing chest X-ray images.")

st.warning(
    "IMPORTANT: This website is an educational project, not a medical "
    "diagnostic tool. Its predictions are experimental and must not be "
    "used to make medical decisions."
)


# -----------------------------
# 2. MODEL SETTINGS
# -----------------------------

MODEL_PATH = "best_pneumonia_resnet18.pth"

# Use the same image size and normalisation as the current training code.
IMAGE_SIZE = 28


# -----------------------------
# 3. CREATE THE RESNET-18 MODEL
# -----------------------------

def create_model():

    # Load ResNet-18 without downloading pretrained weights.
    model = models.resnet18(weights=None)

    # Change the final layer to produce one output:
    # 0 = Normal, 1 = Pneumonia (based on the training labels).
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model


# -----------------------------
# 4. LOAD THE TRAINED MODEL
# -----------------------------

@st.cache_resource
def load_model():

    model = create_model()

    # Load the saved model weights.
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    # Support a few common checkpoint formats.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    # Remove "module." if the model was saved using multiple GPUs.
    state_dict = {
        key.replace("module.", ""): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(state_dict)
    model.eval()

    return model


# -----------------------------
# 5. PREPARE THE UPLOADED IMAGE
# -----------------------------

def preprocess_image(image):

    # Convert the X-ray to grayscale.
    image = image.convert("L")

    # Resize it to match the training dataset.
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Convert the image into a NumPy array.
    image_array = np.array(image).astype(np.float32) / 255.0

    # Normalise pixel values from [0, 1] to [-1, 1].
    image_array = (image_array - 0.5) / 0.5

    # Add three colour channels because ResNet-18 expects RGB input.
    image_array = np.stack([image_array] * 3, axis=0)

    # Convert the array into a PyTorch tensor.
    image_tensor = torch.tensor(image_array, dtype=torch.float32)

    # Add a batch dimension.
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


# -----------------------------
# 6. GENERATE THE GRAD-CAM HEATMAP
# -----------------------------

def generate_gradcam(model, image_tensor, original_image):

    activations = []
    gradients = []

    # Save the feature maps produced by the final ResNet layer.
    def forward_hook(module, input, output):
        activations.append(output)

        # Save gradients during backpropagation.
        output.register_hook(
            lambda gradient: gradients.append(gradient)
        )

    # Attach the hook to the final convolutional block.
    hook = model.layer4.register_forward_hook(forward_hook)

    # Calculate the model output.
    model.zero_grad()

    output = model(image_tensor)

    # Backpropagate the output to calculate Grad-CAM.
    output[0, 0].backward()

    # Remove the hook after collecting the information.
    hook.remove()

    # Get the saved activations and gradients.
    activation = activations[0]
    gradient = gradients[0]

    # Calculate the importance of each feature map.
    weights = gradient.mean(dim=(2, 3), keepdim=True)

    # Combine the feature maps using their importance.
    cam = (weights * activation).sum(dim=1, keepdim=True)

    # Keep only positive values.
    cam = F.relu(cam)

    # Resize the heatmap to match the original image.
    cam = F.interpolate(
        cam,
        size=(original_image.height, original_image.width),
        mode="bilinear",
        align_corners=False
    )

    # Convert the heatmap into a NumPy array.
    cam = cam.squeeze().detach().numpy()

    # Normalise the heatmap between 0 and 1.
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    # Create a red-orange heatmap.
    heatmap = np.zeros(
        (cam.shape[0], cam.shape[1], 3),
        dtype=np.uint8
    )

    heatmap[:, :, 0] = (cam * 255).astype(np.uint8)
    heatmap[:, :, 1] = (cam * 100).astype(np.uint8)

    heatmap_image = Image.fromarray(heatmap)

    # Convert the original image to RGB.
    original_rgb = original_image.convert("RGB")

    # Overlay the heatmap on the original X-ray.
    overlay = Image.blend(
        original_rgb,
        heatmap_image,
        alpha=0.4
    )

    return overlay


# -----------------------------
# 7. IMAGE UPLOAD SECTION
# -----------------------------

st.subheader("Upload a Chest X-ray")

uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Open the uploaded image.
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded chest X-ray",
        width=400
    )

    # Check whether the trained model file exists.
    import os

    if not os.path.exists(MODEL_PATH):

        st.info(
            "The website interface is ready, but the trained model "
            "has not been added yet. Add "
            "`best_pneumonia_resnet18.pth` to the project folder "
            "after training is complete."
        )

    else:

        if st.button("Analyse X-ray"):

            try:

                with st.spinner("Analysing the X-ray..."):

                    # Load the trained model.
                    model = load_model()

                    # Prepare the uploaded image.
                    image_tensor = preprocess_image(image)

                    # Make the prediction.
                    with torch.no_grad():
                        output = model(image_tensor)
                        probability = torch.sigmoid(output).item()

                    # Convert probability into a predicted class.
                    if probability >= 0.5:
                        prediction = "Pneumonia"
                        confidence = probability * 100
                    else:
                        prediction = "Normal"
                        confidence = (1 - probability) * 100

                    # Generate the Grad-CAM visualisation.
                    overlay = generate_gradcam(
                        model,
                        image_tensor,
                        image
                    )

                # -----------------------------
                # 8. DISPLAY THE RESULTS
                # -----------------------------

                st.divider()
                st.subheader("Analysis Results")

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown("### Model Prediction")

                    st.metric(
                        label="Predicted Class",
                        value=prediction
                    )

                    st.metric(
                        label="Model Confidence",
                        value=f"{confidence:.2f}%"
                    )

                with col2:

                    st.markdown("### Grad-CAM Visualisation")

                    st.image(
                        overlay,
                        caption="Red and orange areas indicate regions that influenced the model's prediction.",
                        use_container_width=True
                    )

                st.caption(
                    "The heatmap shows model attention. It does not "
                    "identify confirmed pneumonia or disease locations."
                )

            except Exception as error:

                st.error(
                    f"Something went wrong while analysing the image: {error}"
                )


# -----------------------------
# 9. FOOTER
# -----------------------------

st.divider()

st.caption(
    "PneumoniaScan AI | Educational AI Project | "
    "Not intended for medical diagnosis."
)
```
