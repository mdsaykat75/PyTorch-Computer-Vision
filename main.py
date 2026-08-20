import struct
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# ---------- CONFIG ----------
DATA_DIR = r"D:\PyTorch Computer Vision\data\fashionmnist"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
# ----------------------------

# ---------- LOAD IDX FILES ----------
def load_idx_images(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, 1, rows, cols)
        images = images.astype(np.float32) / 255.0
    return images

def load_idx_labels(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

train_images_path = os.path.join(DATA_DIR, "train-images-idx3-ubyte")
train_labels_path = os.path.join(DATA_DIR, "train-labels-idx1-ubyte")
test_images_path  = os.path.join(DATA_DIR, "t10k-images-idx3-ubyte")
test_labels_path  = os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte")

print("Loading training data...")
X_train = load_idx_images(train_images_path)
y_train = load_idx_labels(train_labels_path)

print("Loading test data...")
X_test = load_idx_images(test_images_path)
y_test = load_idx_labels(test_labels_path)

print(f"Train: {X_train.shape}, Labels: {y_train.shape}")
print(f"Test:  {X_test.shape}, Labels: {y_test.shape}")

X_train_tensor = torch.from_numpy(X_train.copy())
y_train_tensor = torch.from_numpy(y_train.copy()).long()
X_test_tensor  = torch.from_numpy(X_test.copy())
y_test_tensor  = torch.from_numpy(y_test.copy()).long()

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# ---------- MODEL: TinyVGG ----------
class TinyVGG(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_classes):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * 7 * 7, output_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x

model = TinyVGG(input_channels=1, hidden_channels=10, output_classes=10)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(model)
print(f"Device: {DEVICE}")

# ---------- STEP 3: TRAINING LOOP ----------
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{EPOCHS}] completed. Average Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

train_losses = []
test_accuracies = []

for epoch in range(EPOCHS):
    avg_loss = train_one_epoch(epoch)
    accuracy = evaluate()
    train_losses.append(avg_loss)
    test_accuracies.append(accuracy)

print("\nTraining complete!")
print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")

# ---------- STEP 4: VISUALIZATION ----------
import matplotlib
matplotlib.use('Agg')

plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, marker='o', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve - TinyVGG on FashionMNIST')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
plt.close()
print("Saved: loss_curve.png")

model.eval()
with torch.no_grad():
    images, labels = next(iter(test_loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

images = images.cpu()
labels = labels.cpu()
predicted = predicted.cpu()

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    if i >= len(images):
        break
    img = images[i].squeeze(0)
    true_label = CLASS_NAMES[labels[i].item()]
    pred_label = CLASS_NAMES[predicted[i].item()]
    color = 'green' if predicted[i] == labels[i] else 'red'
    ax.imshow(img, cmap='gray')
    ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=9)
    ax.axis('off')

plt.suptitle('Sample Predictions - TinyVGG on FashionMNIST', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: sample_predictions.png")

print("\nAll done! Check 'loss_curve.png' and 'sample_predictions.png'.")
