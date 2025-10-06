import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


# ----------------------------
# 1. Dataset Class (recursive)
# ----------------------------
class AntlerDataset(Dataset):
    def __init__(self, image_root, mask_root, transform=None, exts=(".jpg", ".png", ".jpeg")):
        self.image_root = image_root
        self.mask_root = mask_root
        self.transform = transform
        self.exts = exts

        # Collect all image paths recursively
        self.image_paths = []
        for root, _, files in os.walk(image_root):
            for f in files:
                if f.lower().endswith(exts):
                    self.image_paths.append(os.path.join(root, f))

        # Sort for consistency
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        rel_path = os.path.relpath(img_path, self.image_root)
        mask_path = os.path.join(self.mask_root, rel_path)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for {img_path}, expected {mask_path}")

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = np.array(image)
        mask = np.array(mask)

        # Binarize mask (0 background, 1 antlers)
        mask = (mask > 127).astype(np.int64)

        # Convert to tensors
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        mask = torch.tensor(mask).long()

        return image, mask


# ----------------------------
# 2. Model Setup
# ----------------------------
def get_model(num_classes=2):
    #model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


# ----------------------------
# 3. Training Function
# ----------------------------
def train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)["out"]
                val_loss += criterion(outputs, masks).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model


# ----------------------------
# 4. Inference Function
# ----------------------------
def remove_background(model, image_path, mask_root, image_root, output_path, device):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    input_tensor = torch.tensor(image_np).permute(2,0,1).unsqueeze(0).float() / 255.0
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)["out"]
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]

    result = np.zeros_like(image_np)
    result[pred == 1] = image_np[pred == 1]

    Image.fromarray(result).save(output_path)
    print(f"Saved background-removed image to {output_path}")


# ----------------------------
# 5. Main Script
# ----------------------------
if __name__ == "__main__":
    image_root = "Antlers"   # root folder containing subfolders of trailcam photos
    mask_root = "BAntlers"     # root folder containing matching subfolders of binary masks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & split
    dataset = AntlerDataset(image_root, mask_root)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # Model
    model = get_model(num_classes=2)

    # Train
    model = train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-4)

    # Save trained model
    torch.save(model.state_dict(), "deeplab_antlers.pth")
    print("Model saved as deeplab_antlers.pth")

    # Example inference on one image
    test_image = dataset.image_paths[0]
    remove_background(model, "Test_Image.jpg", mask_root, image_root, "antlers_only.png", device)
