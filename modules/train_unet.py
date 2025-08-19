import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from datasets import load_dataset
import numpy as np
import cv2
import albumentations as A
from pathlib import Path
from PIL import Image

def trees_to_mask(trees, shape=(1024, 1024)):
    mask = np.zeros(shape, dtype=np.uint8)
    for t in trees:
        x, y, r = int(t["x"]), int(t["y"]), int(t["radius"])
        cv2.circle(mask, (x, y), r, 1, -1)
    return mask

class TreesDataset(Dataset):
    def __init__(self, hf_dataset, images_dir, augment=False):
        self.ds = hf_dataset
        self.images_dir = Path(images_dir)
        self.augment = augment
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ]) if augment else None

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        # Correction : charger l'image depuis le chemin dans le champ "image_file" ou similaire
        # Adapte ici le champ si besoin (parfois "image_path" ou "file_name")
        image_filename = item.get("image_file") or item.get("image_path") or item.get("file_name")
        if image_filename is None:
            raise KeyError("Impossible de trouver le champ du nom de fichier image dans l'item du dataset.")
        img_path = self.images_dir / image_filename
        img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        mask = trees_to_mask(item["transformed_trees"], shape=img.shape[:2])
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]
        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

if __name__ == "__main__":
    # Adapter le nom du champ si besoin (vérifie avec print(ds[0].keys()))
    ds = load_dataset("Filipstrozik/satellite_trees_wroclaw_2024")["train"]
    images_dir = "data/satellite_trees_wroclaw_2024/images"  # Adapte ce chemin si besoin

    # Vérification du champ image et affichage d'un exemple
    print("Clés disponibles dans un item du dataset :", ds[0].keys())

    train_dataset = TreesDataset(ds, images_dir=images_dir, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1, activation=None).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    epochs = 10
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}")

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/unet_best.pth")
    print("Modèle sauvegardé dans models/unet_best.pth")
