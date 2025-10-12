import streamlit as st
import torch
from torchvision.models import resnet18, ResNet18_Weights
import cv2
import numpy as np
from PIL import Image
import asyncio
import sys
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.metrics import confusion_matrix
import plotly.express as px

# Try to import SHAP, make it optional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("⚠️ SHAP library not available. Only Grad-CAM explanations will be available.")

if sys.version_info >= (3, 13):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

# --- UI Theme & Animation ---
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered",
)

# --- INSTAGRAM THEME CSS ---
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #405DE6, #5851DB, #833AB4, #C13584, #E1306C, #FD1D1D, #F56040, #F77737, #FCAF45, #FFDC80);
        color: white;
    }
    .stButton>button {
        background-color: #E1306C;
        color: white;
        border-radius: 20px;
        border: 1px solid #E1306C;
    }
    .stFileUploader {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    .stMarkdown, .stSuccess, h1, h3 {
        color: white !important;
    }
    .prediction-card {
        background: rgba(0,0,0,0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .uploaded-image-caption {color: #FFDC80 !important;}
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    <center>
    <lottie-player src="https://assets2.lottiefiles.com/packages/lf20_3u6uxdsq.json" background="transparent" speed="1" style="width: 200px; height: 200px;" loop autoplay></lottie-player>
    </center>
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    """, unsafe_allow_html=True
)

st.title("🧠 Brain Tumor MRI Classifier")
st.markdown(
    "<div style='text-align:center; font-size:1.2em;'>Upload an MRI image and get the predicted tumor type with explainable AI (Grad-CAM or SHAP).</div>", unsafe_allow_html=True
)

# --- Sidebar: Precautions Section ---
st.sidebar.header("🔒 Precautions")
st.sidebar.write("""
- Avoid smoking and excess alcohol consumption.
- Eat a balanced, antioxidant-rich diet with fruits, vegetables, seeds, nuts, and fatty fish.
- Choose whole foods, minimize processed/sugary items, and limit salt.
- Regular moderate exercise and hydration.
- Limit exposure to harmful chemicals and radiation.
- Keep regular medical checkups and report symptoms early.
""")

# --- Sidebar: Food Plan Section ---
st.sidebar.header("🥗 Food Recommendations & Plan")
st.sidebar.write("""
**Recommended Foods:**
- Lean proteins: chicken, turkey, eggs, tofu, legumes
- Healthy fats: avocados, olive oil, seeds, fatty fish (salmon/mackerel)
- Whole grains: brown rice, quinoa, oats, whole wheat
- Colorful fruits and vegetables
- Hydrating foods: cucumber, watermelon, oranges
**To limit/avoid:**
- Processed foods, sugary snacks and drinks, high-sodium and fried food, alcohol
""")

# --- Sidebar: Famous Questions for Chatbot ---
st.sidebar.header("❓ Quick Brain Tumor Questions")
famous_questions = [
    "What are the symptoms of glioma?",
    "How is meningioma different from a pituitary tumor?",
    "What diets are best for brain tumor patients?",
    "What precautions should brain tumor patients follow?",
    "Is exercise safe during brain tumor treatment?"
]
selected_famous = st.sidebar.radio("Click a question to ask:", famous_questions)
if st.sidebar.button("Ask This Question"):
    st.session_state.setdefault('user_question', '')
    st.session_state['user_question'] = selected_famous

# --- Medical AI Chatbot in Sidebar ---
st.sidebar.title("💬 Medical AI Chatbot")
@st.cache_resource
def load_chatbot_model():
    try:
        return pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Chatbot model could not be loaded: {e}")
        return None
chatbot_model = load_chatbot_model()

user_question = st.sidebar.text_area("Ask a medical question about brain tumors or MRI scans:",
                                     value=st.session_state.get('user_question', ''))

if st.sidebar.button("Ask"):
    if not user_question.strip():
        st.sidebar.warning("Please enter a question.")
    elif chatbot_model is None:
        st.sidebar.error("AI chatbot is unavailable.")
    else:
        with st.spinner("AI is thinking..."):
            try:
                prompt = (
                    "<|system|>You are a helpful medical AI assistant. Your purpose is to answer questions "
                    "about brain tumors clearly and factually. Do not invent information. "
                    "Always state that you are an AI and not a substitute for a real doctor.</s><|user|>"
                    f"{user_question}</s><|assistant|>"
                )
                responses = chatbot_model(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
                answer = responses[0]['generated_text'].split("<|assistant|>")[1].strip()
                st.sidebar.markdown(f"**AI Answer:**\n\n{answer}")
            except Exception as e:
                st.sidebar.error(f"Error generating response: {e}")

# --- Model Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()
model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)

try:
    model_path = r"C:\Users\obbin\Desktop\brain tumor project\archive/resnet18_brain_tumor.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()
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

# --- SHAP Explanation ---
def shap_explanation(model, input_tensor, class_names):
    if not SHAP_AVAILABLE:
        return None, None
    try:
        # Background should have shape (batch, channels, height, width)
        background = torch.randn(10, 3, 224, 224).to(input_tensor.device)
        explainer = shap.GradientExplainer(model, background)

        # Make sure input_tensor is also batchified
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Compute SHAP values
        shap_values = explainer.shap_values(input_tensor)

        # For multiclass: shap_values is a list, one array per class
        output = model(input_tensor)
        pred_class = output.argmax(1).item()
        
        # Defensive indexing: only access shap_value for pred_class if exists
        if isinstance(shap_values, list) and len(shap_values) > pred_class:
            class_shap = shap_values[pred_class][0]
        elif isinstance(shap_values, list) and len(shap_values) == 1:
            # Binary/miscase: use first element
            class_shap = shap_values[0][0]
        else:
            # Fallback: use everything
            class_shap = shap_values[0]
        
        shap_gray = np.mean(np.abs(class_shap), axis=0)
        shap_gray = (shap_gray - shap_gray.min()) / (shap_gray.max() - shap_gray.min() + 1e-8)
        return shap_gray, pred_class
    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")
        return None, None


# --- File Upload ---
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded MRI", use_container_width=True)
        image_resized = image.resize((224, 224))
        tensor = transform(image_resized).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            pred_class = output.argmax(1).item()
            pred_label = class_names[pred_class]
            confidence = torch.softmax(output, dim=1)[0, pred_class].item()
        st.markdown(
            f"""
            <div class="prediction-card">
            <h2 style='color:#FFFFFF;'>Prediction</h2>
            <b>Tumor Type:</b> <span style='color:#FCAF45;font-size:1.3em;'>{pred_label.title()}</span><br>
            <b>Confidence:</b> <span style='color:#FFFFFF;'>{confidence*100:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        if SHAP_AVAILABLE:
            explanation_options = ["Grad-CAM", "SHAP"]
        else:
            explanation_options = ["Grad-CAM"]
        explanation_method = st.selectbox(
            "Choose Explanation Method:",
            explanation_options,
            help="Select the XAI method to explain the model's prediction"
        )
        img_np = np.array(image_resized)
        # --- Scatterplot for XAI ---
        st.subheader("Feature Importances Scatterplot (XAI)")
        if explanation_method == "Grad-CAM":
            cam = grad_cam(model, tensor, pred_class)
            cam = cv2.resize(cam, (224, 224))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(img_cv, 0.5, heatmap, 0.5, 0)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            st.markdown("### Model Explanation (Grad-CAM)")
            st.image(overlay_rgb, caption="Grad-CAM Heatmap", use_container_width=True)
            # Grad-CAM feature intensity scatterplot
            cam_flat = cam.flatten()
            coords = np.column_stack(np.unravel_index(np.arange(cam_flat.size), (224, 224)))
            scatter_df = pd.DataFrame({'X': coords[:, 0], 'Y': coords[:, 1], 'Intensity': cam_flat})
            st.scatter_chart(scatter_df, x="X", y="Y", color="Intensity")
        elif explanation_method == "SHAP":
            with st.spinner("Generating SHAP explanation..."):
                shap_gray, _ = shap_explanation(model, tensor, class_names)
            if shap_gray is not None:
                shap_heatmap = cv2.applyColorMap(np.uint8(255 * shap_gray), cv2.COLORMAP_JET)
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                shap_overlay = cv2.addWeighted(img_cv, 0.5, shap_heatmap, 0.5, 0)
                shap_overlay_rgb = cv2.cvtColor(shap_overlay, cv2.COLOR_BGR2RGB)
                st.markdown("### Model Explanation (SHAP)")
                st.image(shap_overlay_rgb, caption="SHAP Attribution Map", use_container_width=True)
                # SHAP scatterplot
                shap_flat = shap_gray.flatten()
                coords = np.column_stack(np.unravel_index(np.arange(shap_flat.size), (224, 224)))
                scatter_df = pd.DataFrame({'X': coords[:, 0], 'Y': coords[:, 1], 'Intensity': shap_flat})
                st.scatter_chart(scatter_df, x="X", y="Y", color="Intensity")
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")

# --- Tumor Type Distribution Graph ---
st.header("Tumor Type Distribution")
try:
    df = pd.read_csv(r"C:\Users\obbin\Desktop\brain tumor project\archive/submission.csv")
    tumor_counts = df['label'].value_counts()
    fig, ax = plt.subplots(facecolor='none')
    tumor_counts.plot(kind='bar', ax=ax, color=['#C13584', '#F56040', '#FCAF45', '#5851DB'])
    ax.set_xlabel("Tumor Type", color='white')
    ax.set_ylabel("Count", color='white')
    ax.set_title("Distribution of Tumor Types in Dataset", color='white')
    ax.grid(axis='y', color='white', alpha=0.3)
    ax.tick_params(colors='white')
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Could not load tumor distribution data: {e}")

# --- Model Performance Analysis ---
st.title("📊 Model Performance Analysis")
st.write("This page shows how accurately the model performs on the entire test dataset.")

from torchvision import datasets
TEST_DATA_PATH = r"C:\Users\obbin\Desktop\brain tumor project\archive/Testing"

@st.cache_data
def evaluate_model_on_test_set(test_dir):
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    true_labels = []
    pred_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())
    return true_labels, pred_labels

try:
    true_labels, pred_labels = evaluate_model_on_test_set(TEST_DATA_PATH)
    st.subheader("Prediction Accuracy on Test Dataset")
    st.write("The confusion matrix below shows the model's predictions (x-axis) versus the actual true labels (y-axis). The diagonal values represent correct predictions.")
    cm = confusion_matrix(true_labels, pred_labels)
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted Class", y="True Class", color="Count"),
        x=class_names,
        y=class_names,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix"
    )
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig, use_container_width=True)
    from sklearn.metrics import classification_report
    report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.subheader("Classification Report")
    st.dataframe(df_report)
except FileNotFoundError:
    st.error(f"Test data directory not found at '{TEST_DATA_PATH}'. Please provide the correct path to your 'Testing' folder.")
except Exception as e:
    st.error(f"An error occurred during evaluation: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("<small>⚠️ This chatbot is for informational purposes only and does not provide medical advice. Always consult a healthcare professional.</small>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<div style='text-align:center;'>Made with ❤️ using <b>Streamlit</b> & <b>PyTorch</b></div> <b>By Haswanth and Dinakar</b>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;'><small>© 2024 Brain Tumor Classifier. All rights reserved.</small></div>", unsafe_allow_html=True)
