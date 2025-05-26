import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ Settings
val_dir = '/content/drive/MyDrive/Multi-Crop Plant Leaf Nutrient Deficiency Classification/data/val'
model_save_dir = '/content/drive/MyDrive/Multi-Crop Plant Leaf Nutrient Deficiency Classification/models/'
output_dir = '/content/drive/MyDrive/Multi-Crop Plant Leaf Nutrient Deficiency Classification/outputs/'
os.makedirs(output_dir, exist_ok=True)

batch_size = 32
image_size = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ Transforms for validation
val_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ✅ Validation data
val_dataset = ImageFolder(val_dir, transform=val_transforms)
class_names = val_dataset.classes
num_classes = len(class_names)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ✅ Load individual models
def load_model(model_name, num_classes):
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    
    model.load_state_dict(torch.load(os.path.join(model_save_dir, f'best_model_{model_name}.pth'), map_location=device))
    model = model.to(device)
    model.eval()
    return model

models_list = ['efficientnet_b0', 'resnet50', 'densenet121']
loaded_models = [load_model(name, num_classes) for name in models_list]

# ✅ Ensemble prediction
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = [F.softmax(model(inputs), dim=1) for model in loaded_models]
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        _, preds = torch.max(avg_output, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ✅ Evaluation
acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\n✅ Ensemble Accuracy: {acc:.4f}")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Ensemble Model')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix_ensemble.png'))
plt.show()

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

print("\n✅ Ensemble Predictions Completed!")
