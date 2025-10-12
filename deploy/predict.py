import torch
from torchvision.models import resnet18, ResNet18_Weights
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

# Setup model
model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)  # 4 classes
model.load_state_dict(torch.load(r"C:\Users\obbin\Desktop\brain tumor project\archive\resnet18_brain_tumor.pth", map_location=device))
model = model.to(device)
model.eval()

class_names = ["glioma", "meningioma", "notumor", "pituitary"]  # Update if needed

def predict_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred_class = output.argmax(1).item()
    return class_names[pred_class]

# Example usage:
if __name__ == "__main__":
    img_path = r"C:\Users\obbin\Desktop\brain tumor project\archive\Testing\notumor\your_image.jpg"  # Change to your test image
    result = predict_image(img_path)
    print("Predicted tumor type:", result)