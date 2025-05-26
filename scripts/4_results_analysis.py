# 📄 File: results_analysis.py

import os
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ✅ Paths and setup
val_dir = '/content/drive/MyDrive/Multi-Crop Plant Leaf Nutrient Deficiency Classification/data/val'
model_dir = '/content/drive/MyDrive/Multi-Crop Plant Leaf Nutrient Deficiency Classification/models/'
output_dir = '/content/drive/MyDrive/Multi-Crop Plant Leaf Nutrient Deficiency Classification/outputs/'
os.makedirs(output_dir, exist_ok=True)

batch_size = 32
image_size = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ Transforms
val_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ✅ Load dataset
val_dataset = ImageFolder(val_dir, transform=val_transforms)
class_names = val_dataset.classes
num_classes = len(class_names)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ✅ Model loader
def load_model(name):
    if name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif name == 'densenet121':
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(torch.load(os.path.join(model_dir, f'best_model_{name}.pth'), map_location=device))
    model = model.to(device)
    model.eval()
    return model

# ✅ Evaluate a model
def evaluate_model(model, name):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(F.softmax(outputs, dim=1), 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{name}.png'))
    plt.close()

    print(f"\n✅ Classification Report for {name}:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return name, acc, report['weighted avg']['precision'], report['weighted avg']['recall'], report['weighted avg']['f1-score']

# ✅ Evaluate all individual models
models_to_eval = ['efficientnet_b0', 'resnet50', 'densenet121']
summary = []

for model_name in models_to_eval:
    model = load_model(model_name)
    row = evaluate_model(model, model_name)
    summary.append(row)

# ✅ Optional: Evaluate Ensemble
print("\n✅ Evaluating Soft Voting Ensemble...")
all_preds, all_labels = [], []
models_loaded = [load_model(m) for m in models_to_eval]

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = [F.softmax(model(inputs), dim=1) for model in models_loaded]
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        _, preds = torch.max(avg_output, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
cm = confusion_matrix(all_labels, all_preds)

# Save Ensemble confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Ensemble')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix_ensemble.png'))
plt.close()

print("\n✅ Classification Report for Ensemble:")
print(classification_report(all_labels, all_preds, target_names=class_names))

summary.append(("ensemble", acc, report['weighted avg']['precision'], report['weighted avg']['recall'], report['weighted avg']['f1-score']))

# ✅ Save summary table
df_summary = pd.DataFrame(summary, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
df_summary.to_csv(os.path.join(output_dir, 'model_comparison_summary.csv'), index=False)
print("\n📊 Summary Table:\n")
print(df_summary)
print("\n✅ Evaluation Complete. Results saved to outputs/")
