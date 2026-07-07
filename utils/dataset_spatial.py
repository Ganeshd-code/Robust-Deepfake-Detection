import pandas as pd
import cv2
import torch

from torch.utils.data import Dataset


class DeepfakeDataset(Dataset):

    def __init__(
        self,
        csv_file,
        transform=None
    ):

        self.data = pd.read_csv(csv_file)

        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        path = self.data.iloc[idx]["path"]
        label = self.data.iloc[idx]["label"]

        # ==================================================
        # Load Image
        # ==================================================
        img = cv2.imread(path)

        if img is None:

            raise ValueError(
                f"Image not found: {path}"
            )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # ==================================================
        # Transform
        # ==================================================
        if self.transform:

            img = self.transform(img)

        return (
            img,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )