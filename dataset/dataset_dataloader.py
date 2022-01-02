import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, metadata_df, transform=None, default_image_size=400):
        self.metadata_df = metadata_df
        self.transform = transform
        self.class_names = sorted(self.metadata_df['label_name'].unique().tolist())
        self.default_transform = transforms.Compose([
            transforms.Resize((default_image_size, default_image_size)),
            transforms.ToTensor()
        ])
        print('`CustomDataset` created')
        print('class amount: ', len(self.class_names), 'class names: ', self.class_names)
        print('image amount: ', len(self.metadata_df))

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        df_row = self.metadata_df.iloc[idx]
        image_path = df_row[0]
        label = df_row[1]
        label_name = df_row[2]

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        else:
            image = self.default_transform(image)

        metadata = np.array([label])
        metadata = metadata.astype('int').reshape(-1)

        return image, metadata, label_name


def create_dataset_dataloader(metadata_df, stratify_column='label', image_size=400, test_size=0.2, batch_size=32):
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    valid_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_df, valid_df = train_test_split(metadata_df, test_size=test_size, stratify=metadata_df[stratify_column])

    train_dataset = CustomDataset(train_df, transform=train_transforms)
    valid_dataset = CustomDataset(valid_df, transform=valid_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=2, shuffle=True)

    return train_dataset, valid_dataset, train_loader, valid_loader


# function demo
if __name__ == '__main__':
    csv_path = 'pokemon_dataset.csv'
    metadata_df = pd.read_csv(csv_path)
    train_dataset, valid_dataset, train_loader, valid_loader = create_dataset_dataloader(metadata_df)
