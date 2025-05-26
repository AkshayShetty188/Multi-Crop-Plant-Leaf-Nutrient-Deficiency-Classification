import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# ✅ Settings
train_dir = '/content/data/train'
val_dir = '/content/data/val'
# train_dir = '/content/drive/MyDrive/Multi-Crop Plant Leaf Nutrient Deficiency Classification/data/train'
# val_dir = '/content/drive/MyDrive/Multi-Crop Plant Leaf Nutrient Deficiency Classification/data/val'
model_save_dir = '/content/drive/MyDrive/Multi-Crop Plant Leaf Nutrient Deficiency Classification/models/'
os.makedirs(model_save_dir, exist_ok=True)

batch_size = 32
num_epochs = 24
image_size = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ Transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ✅ Datasets and loaders
train_dataset = ImageFolder(train_dir, transform=train_transforms)
val_dataset = ImageFolder(val_dir, transform=val_transforms)
class_names = train_dataset.classes
num_classes = len(class_names)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
dataloaders = {'train': train_loader, 'val': val_loader}

# ✅ Models
model_list = {
    'efficientnet_b0': models.efficientnet_b0(weights='DEFAULT'),
    'resnet50': models.resnet50(weights='DEFAULT'),
    'densenet121': models.densenet121(weights='DEFAULT')
}

# ✅ Helper functions
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ✅ Training function
def train_model(model, model_name, use_mixup=False):
    print(f"\nTraining {model_name}...")

    # Replace classifier
    if model_name.startswith('efficientnet'):
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name.startswith('resnet'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.startswith('densenet'):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_acc = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, correct, total = 0.0, 0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                if phase == 'train' and use_mixup:
                    inputs, y_a, y_b, lam = mixup_data(inputs, labels)
                    outputs = model(inputs)
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                    _, preds = torch.max(outputs, 1)
                    correct += (lam * preds.eq(y_a).sum() + (1 - lam) * preds.eq(y_b).sum())
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    correct += torch.sum(preds == labels.data)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = correct.double() / total
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val':
                scheduler.step(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), os.path.join(model_save_dir, f"best_model_{model_name}.pth"))

    return history

# ✅ Training loop
histories = {}
for name, model in model_list.items():
    use_mixup = True if name.startswith('efficientnet') else False
    history = train_model(model, name, use_mixup)
    histories[name] = history
# for name, model in {'densenet121': model_list['densenet121']}.items():
#     use_mixup = False
#     history = train_model(model, name, use_mixup)
#     histories[name] = history



# ✅ Save training curves
def plot_curves(history, model_name):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title(f"Loss Curve - {model_name}")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.title(f"Accuracy Curve - {model_name}")
    plt.legend()

    os.makedirs('/content/outputs', exist_ok=True)
    plt.savefig(f"/content/outputs/{model_name}_training_curves.png")
    plt.show()

for name in histories:
    plot_curves(histories[name], name)

print("\n✅ All Models Trained and Saved Successfully!")