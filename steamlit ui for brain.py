import streamlit as st
import torch
from torchvision.models import resnet18, ResNet18_Weights
import cv2
import numpy as np
from PIL import Image
import openai
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt

# --- UI Theme ---
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered",
)

st.markdown(
"""
    <style>
    .main {background-color: #f5f7fa;}
    .stButton>button {background-color: #4F8BF9; color: white;}
    .stFileUploader {background-color: #e3eafc;}
    .stMarkdown {font-size: 1.1em;}
    .uploaded-image-caption {color: #4F8BF9 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("Upload an MRI image and get the predicted tumor type with a visual explanation (Grad-CAM).")

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
    st.image(image, caption="Uploaded MRI", use_column_width=True)
    image_resized = image.resize((224, 224))  # Use PIL resize
    tensor = transform(image_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred_class = output.argmax(1).item()
        pred_label = class_names[pred_class]
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()

    st.success(f"**Predicted Tumor Type:** {pred_label.title()} ({confidence*100:.2f}% confidence)")

    # Grad-CAM Visualization
    img_np = np.array(image_resized)
    cam = grad_cam(model, tensor, pred_class)
    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

    st.markdown("### Model Explanation (Grad-CAM)")
    st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)

    st.markdown(
        """<div style='background-color: black; padding: 10px; border-radius: 10px;'>
        <b>What is Grad-CAM?</b><br>
        Grad-CAM highlights regions in the MRI that most influenced the model's prediction.<br>
        <i>Red/yellow areas are most important for the decision.</i>
        </div>
        """,
        unsafe_allow_html=True
        
    )
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and PyTorch.")

# --- Tumor Type Distribution Graph ---
st.header("Tumor Type Distribution")

# Load submission.csv and calculate tumor type counts
df = pd.read_csv("submission.csv")
tumor_counts = df['label'].value_counts()

# Plot bar chart
fig, ax = plt.subplots()
tumor_counts.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_xlabel("Tumor Type")
ax.set_ylabel("Count")
ax.set_title("Distribution of Tumor Types in Dataset")
ax.grid(axis='y')

st.pyplot(fig)

# --- Medical AI Chatbot in Sidebar ---
st.sidebar.title("💬 Medical AI Chatbot (Gemini)")

gemini_api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password")

if gemini_api_key:
    user_question = st.sidebar.text_area("Ask a medical question about brain tumors or MRI scans:")

    if st.sidebar.button("Ask"):
        with st.spinner("AI is thinking..."):
            try:
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel("models/gemini-pro")
                prompt = (
                    "You are a helpful medical assistant specialized in brain tumors and MRI interpretation. "
                    "Always remind users that your advice is not a substitute for a real doctor's consultation.\n\n"
                    f"User: {user_question}"
                )
                response = model.generate_content(prompt)
                answer = response.text
                st.sidebar.markdown(f"**AI Answer:**\n\n{answer}")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
else:
    st.sidebar.info("Enter your Gemini API key to use the chatbot.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>⚠️ This chatbot is for informational purposes only and does not provide medical advice. Always consult a healthcare professional.</small>",
    unsafe_allow_html=True
)

