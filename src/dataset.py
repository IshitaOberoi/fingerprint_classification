import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FingerprintDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform

        csv_path = os.path.join(folder_path, "_classes.csv")
        self.df = pd.read_csv(csv_path)

        # filename column is first, label column is second
        self.image_col = self.df.columns[0]
        self.label_col = self.df.columns[1]

        self.classes = sorted(self.df[self.label_col].unique())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = row[self.image_col]
        label_name = row[self.label_col]

        img_path = os.path.join(self.folder_path, img_name)
        image = Image.open(img_path).convert("RGB")

        label = self.classes.index(label_name)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_ds = FingerprintDataset(
        os.path.join(data_dir, "train"), transform
    )
    val_ds = FingerprintDataset(
        os.path.join(data_dir, "valid"), transform
    )
    test_ds = FingerprintDataset(
        os.path.join(data_dir, "test"), transform
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_ds.classes
