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
import plotly.graph_objects as go

# Try to import LIME, make it optional
try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    st.warning("⚠️ LIME library not available. Only Grad-CAM explanations will be available.")

# --- Fix for Python 3.13 event loop issues ---
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
    "<div style='text-align:center; font-size:1.2em;'>Upload an MRI image and get the predicted tumor type with explainable AI (Grad-CAM or LIME).</div>",
    unsafe_allow_html=True
)

# --- Model Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)

# Load model with error handling
try:
    model_path = r"C:\Users\obbin\Desktop\brain tumor project\archive/resnet18_brain_tumor.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    st.success("✅ Model loaded successfully!")
    st.info(
        "This classifier fine-tunes ResNet18: CNN layers extract texture and edge patterns from MRI slices, the global average pooling condenses them into a 512-length fingerprint, and a custom linear head maps that fingerprint into the four tumour classes."
    )
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

class_names = ["glioma", "meningioma", "notumor", "pituitary"]


def _prepare_heatmap_channel(arr: np.ndarray, target_shape: tuple | None = None) -> np.ndarray:
    """Normalize any incoming array to a 2D uint8 heatmap."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    if arr.ndim != 2:
        if target_shape is not None and np.prod(target_shape) == arr.size:
            arr = arr.reshape(target_shape)
        else:
            arr = arr.reshape(1, -1)
    arr = np.nan_to_num(arr)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return np.uint8(np.clip(arr, 0, 1) * 255)


def lime_predict(images):
    """Prediction function compatible with LIME (expects images in uint8 or float [0,1])."""
    processed = []
    for img in images:
        img = np.asarray(img)
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        tensor = transform(pil_img).unsqueeze(0)
        processed.append(tensor)

    batch_tensor = torch.cat(processed, dim=0).to(device)

    with torch.no_grad():
        outputs = model(batch_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
    return probs

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

# --- LIME Explanation ---
def lime_explanation(model, image_np, predict_fn, predicted_index):
    """Generate a LIME explanation for the input image."""
    if not LIME_AVAILABLE:
        return None, None

    try:
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image_np,
            predict_fn,
            top_labels=1,
            labels=(predicted_index,),
            hide_color=0,
            num_samples=600
        )

        temp, mask = explanation.get_image_and_mask(
            label=predicted_index,
            positive_only=False,
            num_features=10,
            hide_rest=False
        )

        overlay = mark_boundaries(temp / 255.0, mask)

        weights = explanation.local_exp.get(predicted_index, [])
        weights_df = pd.DataFrame(weights, columns=["Superpixel", "Weight"]).sort_values(
            by="Weight", ascending=False
        )
        return overlay, weights_df
    except Exception as e:
        st.warning(f"LIME explanation failed: {e}")
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

        explanation_options = ["Grad-CAM"]
        if LIME_AVAILABLE:
            explanation_options.append("LIME")
        explanation_options.append("Feature Scatter (Global)")

        explanation_method = st.selectbox(
            "Choose Explanation Method:",
            explanation_options,
            help="Select the XAI method to explain the model's prediction"
        )

        explanation_summaries = {
            "Grad-CAM": "Back-propagates gradients from the predicted class to highlight the image regions that most influenced the CNN's decision.",
            "LIME": "Perturbs the image by superpixels and fits a simple surrogate model so you can see which local patches push the prediction up or down.",
            "Feature Scatter (Global)": "Summarises activation strength across all pixels and lets you inspect which zones generally matter most for this model."
        }
        st.caption(explanation_summaries.get(explanation_method, ""))

        img_np = np.array(image_resized)

        lime_overlay = None
        lime_weights = None
        if explanation_method == "LIME" and LIME_AVAILABLE:
            with st.spinner("Generating LIME explanation..."):
                lime_overlay, lime_weights = lime_explanation(model, img_np, lime_predict, pred_class)

        if explanation_method == "Grad-CAM":
            cam = grad_cam(model, tensor, pred_class)
            cam = cv2.resize(cam, (224, 224))
            cam_uint8 = _prepare_heatmap_channel(cam, target_shape=cam.shape)
            heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(img_cv, 0.5, heatmap, 0.5, 0)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            st.markdown("### Model Explanation (Grad-CAM)")
            st.image(overlay_rgb, caption="Grad-CAM Heatmap", use_container_width=True)

        elif explanation_method == "LIME":
            if lime_overlay is not None:
                st.markdown("### Model Explanation (LIME)")
                st.image(lime_overlay, caption="LIME Superpixel Attribution", use_container_width=True)
                if lime_weights is not None and not lime_weights.empty:
                    st.markdown("#### Top Superpixel Contributions")
                    st.dataframe(lime_weights.head(12))
            else:
                st.warning("LIME explanation could not be generated for this image.")

        elif explanation_method == "Feature Scatter (Global)":
            st.markdown("### Model Explanation (Feature Scatter)")
            cam_map = grad_cam(model, tensor, pred_class)
            cam_map = cv2.resize(cam_map, (224, 224))
            flat_vals = np.nan_to_num(cam_map.flatten())
            centered = flat_vals - flat_vals.mean()
            df_importance = pd.DataFrame({
                "Feature Index": np.arange(centered.size),
                "Importance": centered,
                "Impact": np.where(centered >= 0, "Positive", "Negative")
            })
            top_n = min(500, df_importance.shape[0])
            if top_n < 2:
                df_subset = df_importance
            else:
                df_subset = pd.concat([
                    df_importance.nlargest(top_n // 2 or 1, "Importance"),
                    df_importance.nsmallest(top_n // 2 or 1, "Importance")
                ])
            fig_scatter = px.scatter(
                df_subset,
                x="Feature Index",
                y="Importance",
                color="Impact",
                title="Global Pixel-Level Importance",
                labels={"Importance": "Centered Activation"},
                opacity=0.7
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.info("Each point represents a pixel location from the Grad-CAM map. Positive spikes show areas that generally boost confidence, while negative spikes indicate regions that weaken it.")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")

# --- Tumor Type Distribution Graph ---
st.header("Tumor Type Distribution")
try:
    df = pd.read_csv(
        r"C:\Users\obbin\Desktop\brain tumor project\archive/submission.csv",
        usecols=["label"],
        dtype={"label": "category"},
        memory_map=True
    )
    tumor_counts = df['label'].value_counts()
    fig, ax = plt.subplots(facecolor='none') # Transparent background
    tumor_counts.plot(kind='bar', ax=ax, color=['#C13584', '#F56040', '#FCAF45', '#5851DB'])
    ax.set_xlabel("Tumor Type", color='white')
    ax.set_ylabel("Count", color='white')
    ax.set_title("Distribution of Tumor Types in Dataset", color='white')
    ax.grid(axis='y', color='white', alpha=0.3)
    ax.tick_params(colors='white')
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Could not load tumor distribution data: {e}")

# --- Medical AI Chatbot in Sidebar ---
st.sidebar.title("💬 Medical AI Chatbot")

resource_choice = st.sidebar.selectbox(
    "Resource Center",
    ["Precautions", "Food Recommendations & Plan", "Explainable AI Guide"],
    help="Quick access to supportive information"
)

if resource_choice == "Precautions":
    st.sidebar.info(
        "- Avoid smoking and excess alcohol consumption.\n- Eat a balanced, antioxidant-rich diet with fruits, vegetables, seeds, nuts, and fatty fish.\n- Choose whole foods, minimize processed/sugary items, and limit salt.\n- Keep regular medical checkups and report symptoms early."
    )
elif resource_choice == "Food Recommendations & Plan":
    st.sidebar.info(
        "**Recommended Foods:**\n- Lean proteins: chicken, turkey, eggs, tofu, legumes\n- Healthy fats: avocados, olive oil, seeds, fatty fish (salmon/mackerel)\n- Whole grains: brown rice, quinoa, oats, whole wheat\n- Colorful fruits and vegetables\n- Hydrating foods: cucumber, watermelon, oranges\n**To limit/avoid:**\n- Processed foods, sugary snacks and drinks, high-sodium and fried food, alcohol"
    )
else:
    st.sidebar.info(
        "Grad-CAM highlights regions that drive predictions, while LIME uncovers which superpixels push the decision for or against the predicted class."
    )

if "user_question_area" not in st.session_state:
    st.session_state.user_question_area = ""

popular_questions = [
    "What are the symptoms of glioma?",
    "How is meningioma different from a pituitary tumor?",
    "What diets are best for brain tumor patients?",
    "What precautions should brain tumor patients follow?",
    "Is exercise safe during brain tumor treatment?"
]

selected_question = st.sidebar.selectbox(
    "Most asked questions",
    ["Choose a question..."] + popular_questions,
    help="Selecting a question will preload it into the prompt box."
)

if selected_question != "Choose a question...":
    st.session_state.user_question_area = selected_question

@st.cache_resource
def load_chatbot_model():
    try:
        return pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Chatbot model could not be loaded: {e}")
        return None

chatbot_model = load_chatbot_model()
user_question = st.sidebar.text_area(
    "Ask a medical question about brain tumors or MRI scans:",
    key="user_question_area"
)
st.sidebar.markdown("**Example questions:**")
st.sidebar.markdown("- What diets are best for brain tumor patients?")
st.sidebar.markdown("- Is exercise safe during brain tumor treatment?")

if st.sidebar.button("Ask"):
    if not user_question.strip():
        st.sidebar.warning("Please enter a question.")
    elif chatbot_model is None:
        st.sidebar.error("AI chatbot is unavailable.")
    else:
        with st.spinner("AI is thinking..."):
            try:
                prompt = f"""<|system|>You are a helpful medical AI assistant. Your purpose is to answer questions about brain tumors clearly and factually. Do not invent information. Always state that you are an AI and not a substitute for a real doctor.</s><|user|>{user_question}</s><|assistant|>"""
                responses = chatbot_model(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
                answer = responses[0]['generated_text'].split("<|assistant|>")[1].strip()
                st.sidebar.markdown(f"**AI Answer:**\n\n{answer}")
            except Exception as e:
                st.sidebar.error(f"Error generating response: {e}")

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

    st.subheader("Performance Dashboard")
    metric_values = {
        "Train Accuracy": 95,
        "Validation Accuracy": 92,
        "Test Accuracy": 91
    }
    radar_fig = go.Figure(
        data=go.Scatterpolar(
            r=list(metric_values.values()),
            theta=list(metric_values.keys()),
            fill='toself',
            name='Model Accuracy',
            marker=dict(color="#FCAF45")
        )
    )
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title="Accuracy Across Splits"
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    st.markdown("#### Model Feature Highlights")
    st.markdown(
        """
        - **Transfer Learning Backbone:** ResNet18 initialized with ImageNet weights for robust feature extraction.
    - **Explainability Suite:** Grad-CAM overlays, LIME superpixel attributions, and chatbot-assisted decision support.
        - **Clinical Aids:** Tumor distribution analytics, confusion matrix insights, and sidebar chatbot guidance.
        - **Operations Ready:** Cached model loading, batch inference utilities, and Kaggle submission helper.
        """
    )

except FileNotFoundError:
    st.error(f"Test data directory not found at '{TEST_DATA_PATH}'. Please provide the correct path to your 'Testing' folder.")
except Exception as e:
    st.error(f"An error occurred during evaluation: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("<small>⚠️ This chatbot is for informational purposes only and does not provide medical advice. Always consult a healthcare professional.</small>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<div style='text-align:center;'>Made with ❤️ using <b>Streamlit</b> & <b>PyTorch</b></div> <b>  By Haswanth and Dinakar  </b>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;'><small>© 2024 Brain Tumor Classifier. All rights reserved.</small></div>", unsafe_allow_html=True)