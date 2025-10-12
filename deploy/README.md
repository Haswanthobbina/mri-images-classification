# 🧠 Brain Tumor Classification with ResNet18 and Grad-CAM (PyTorch)

This project uses **PyTorch** and **transfer learning with ResNet18** to classify brain MRI scans into four tumor categories. It also integrates **Grad-CAM visualizations** to highlight which parts of the image most influenced the model’s prediction.

---

## 🧪 Classes

The model is trained to classify MRI scans into:

- 🧠 Glioma
- 🧠 Meningioma
- 🧠 Pituitary Tumor
- ✅ No Tumor

---

## 🧠 Built With

- `PyTorch` — for deep learning training and model definition
- `torchvision.models.ResNet18` — pretrained on ImageNet, fine-tuned
- `Grad-CAM` — for explainable AI
- `Matplotlib` & `OpenCV` — for visualization
- `Kaggle` — as the primary training environment (but fully portable to local)

---

## 📂 Dataset

**Source**: [Brain MRI Images for Brain Tumor Detection (Kaggle)](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

### Folder Structure
```
/Training/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/

/Testing/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

## 🛠️ Installation (for local use)

```bash
pip install torch torchvision matplotlib opencv-python

```
## 🖼️ Grad-CAM Example Output

![Grad-CAM Heatmap](https://github.com/user-attachments/assets/f08b0d3a-a53e-4810-91d4-308f283f8d3c)


---

## 📦 Features

| Feature                     | ✅ Status |
|-----------------------------|----------|
| ResNet18 Transfer Learning  | ✅ Yes    |
| 20 Epoch Training           | ✅ Yes    |
| Model Accuracy Evaluation   | ✅ Yes    |
| Grad-CAM Visualization      | ✅ Yes    |
| Model Saving & Loading      | ✅ Yes    |
| Kaggle-Compatible Notebook  | ✅ Yes    |

---

## 🛠️ Setup (Local)

```bash
pip install torch torchvision matplotlib opencv-python

```
| Metric         | Value |
| -------------- | ----- |
| Train Accuracy | \~95% |
| Val Accuracy   | \~92% |
| Test Accuracy  | \~91% |


