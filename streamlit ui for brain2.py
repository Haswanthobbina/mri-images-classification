import streamlit as st
import torch
from torchvision.models import resnet18, ResNet18_Weights
import cv2
import numpy as np
from PIL import Image

# --- UI Theme & Animation ---
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered",
)

# Add Lottie animation (3D style effect)
st.markdown(
    """
    <style>
    .main {background: linear-gradient(135deg, #e3eafc 0%, #f5f7fa 100%);}
    .stButton>button {background-color: #4F8BF9; color: white;}
    .stFileUploader {background-color: #e3eafc;}
    .stMarkdown {font-size: 1.1em;}
    .prediction-card {
        background: rgba(255,255,255,0.8);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Lottie animation (brain scan 3D effect)
st.markdown(
    """
    <center>
    <lottie-player src="https://assets2.lottiefiles.com/packages/lf20_3u6uxdsq.json" background="transparent" speed="1" style="width: 200px; height: 200px;" loop autoplay></lottie-player>
    </center>
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Brain Tumor MRI Classifier")
st.markdown(
    "<div style='text-align:center; font-size:1.2em;'>Upload an MRI image and get the predicted tumor type with a visual explanation (Grad-CAM).</div>",
    unsafe_allow_html=True
)

# --- Model Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)
model.load_state_dict(torch.load(r"C:\Users\obbin\Desktop\brain tumor project\archive\resnet18_brain_tumor.pth", map_location=device))
model = model.to(device)
model.eval()

class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# --- Grad-CAM ---
def grad_cam(model, input_tensor, target_class):
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    hook_f = model.layer4.register_forward_hook(forward_hook)
    hook_b = model.layer4.register_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()

    hook_f.remove()
    hook_b.remove()

    grads = gradients[0].squeeze(0)
    acts = activations[0].squeeze(0)
    weights = grads.mean(dim=[1, 2])
    cam = torch.zeros(acts.shape[1:], dtype=torch.float32).to(device)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = cam.cpu().detach().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / cam.max() if cam.max() != 0 else cam
    return cam

# --- File Upload ---
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded MRI", use_column_width=True)
    image_resized = image.resize((224, 224))  # Use PIL resize
    tensor = transform(image_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred_class = output.argmax(1).item()
        pred_label = class_names[pred_class]
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()

    st.markdown(
        f"""
        <div class="prediction-card">
        <h2 style='color:#4F8BF9;'>Prediction</h2>
        <b>Tumor Type:</b> <span style='color:#d7263d;font-size:1.3em;'>{pred_label.title()}</span><br>
        <b>Confidence:</b> <span style='color:#1b998b;'>{confidence*100:.2f}%</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Grad-CAM Visualization
    img_np = np.array(image_resized)
    cam = grad_cam(model, tensor, pred_class)
    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

    st.markdown("### Model Explanation (Grad-CAM)")
    st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)

    st.markdown(
        """
        <div style='background-color: #e4eafc; padding: 10px; border-radius: 10px;'>
        <b>What is Grad-CAM?</b><br>
        Grad-CAM highlights regions in the MRI that most influenced the model's prediction.<br>
        <i>Red/yellow areas are most important for the decision.</i>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(
    "<div style='text-align:center;'>Made with ❤️ using <b>Streamlit</b> & <b>PyTorch</b></div>",
    unsafe_allow_html=True
)