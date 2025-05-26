import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image

# ✅ Settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = '/content/drive/MyDrive/Multi-Crop Plant Leaf Nutrient Deficiency Classification/data/val'
model_save_dir = '/content/drive/MyDrive/Multi-Crop Plant Leaf Nutrient Deficiency Classification/models/'
output_dir = '/content/drive/MyDrive/Multi-Crop Plant Leaf Nutrient Deficiency Classification/outputs/'
os.makedirs(output_dir, exist_ok=True)

image_size = 224
batch_size = 1

# ✅ Transforms
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ✅ Load dataset
val_dataset = ImageFolder(data_dir, transform=transform)
class_names = val_dataset.classes
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# ✅ GradCAM Helper
def get_gradcam(img_tensor, model, target_layer):
    gradients = []
    activations = []

    def save_gradients_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activations_hook(module, input, output):
        activations.append(output)

    hook1 = target_layer.register_forward_hook(save_activations_hook)
    hook2 = target_layer.register_backward_hook(save_gradients_hook)

    output = model(img_tensor)
    pred_class = output.argmax().item()
    loss = output[0, pred_class]
    model.zero_grad()
    loss.backward()

    gradients_ = gradients[0].cpu().data.numpy()[0]
    activations_ = activations[0].cpu().data.numpy()[0]

    weights = np.mean(gradients_, axis=(1,2))
    cam = np.zeros(activations_.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activations_[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (image_size, image_size))
    cam -= np.min(cam)
    cam /= np.max(cam)

    hook1.remove()
    hook2.remove()

    return cam, pred_class

# ✅ Loop through all 3 models
model_names = ['efficientnet_b0', 'resnet50', 'densenet121']

for model_name in model_names:
    print(f"\n🔍 Generating GradCAM for: {model_name}")

    # Load model
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
        target_layer = model.features[-1]
    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
        target_layer = model.layer4[-1]
    elif model_name == 'densenet121':
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, len(class_names))
        target_layer = model.features[-1]

    model.load_state_dict(torch.load(os.path.join(model_save_dir, f'best_model_{model_name}.pth'), map_location=device))
    model = model.to(device)
    model.eval()

    # Visualize 5 random images
    for idx, (inputs, labels) in enumerate(val_loader):
        if idx >= 5:
            break

        inputs = inputs.to(device)
        cam, pred_class = get_gradcam(inputs, model, target_layer)

        img = inputs.cpu().squeeze().permute(1,2,0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[..., ::-1]  # BGR to RGB
        final_img = heatmap * 0.4 + img * 0.6

        plt.figure(figsize=(6,6))
        plt.imshow(final_img)
        plt.title(f"{model_name} | True: {class_names[labels.item()]} | Pred: {class_names[pred_class]}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'gradcam_{model_name}_{idx}.png'))
        plt.close()

print("\n✅ GradCAM Visualizations Completed for All Models!")
