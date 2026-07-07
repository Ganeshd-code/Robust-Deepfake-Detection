import pandas as pd
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

from utils.fft import get_fft


class AsymmetricDataset(Dataset):

    def __init__(
        self,
        csv,
        t_spatial,
        t_freq
    ):

        self.data = pd.read_csv(csv)

        self.t_s = t_spatial
        self.t_f = t_freq

        # ==================================================
        # FFT Tensor Transform
        # deterministic normalization
        # ==================================================
        self.fft_tensor_transform = transforms.Compose([

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

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
        # Spatial Branch
        # cleaner branch
        # ==================================================
        spatial = self.t_s(img)

        # ==================================================
        # Frequency Branch
        # augmentation BEFORE FFT
        # ==================================================

        # apply frequency augmentations
        distorted = self.t_f(img)

        # ==================================================
        # Tensor -> Numpy Image
        # ==================================================
        distorted = distorted.permute(
            1,
            2,
            0
        ).cpu().numpy()

        # ==================================================
        # Denormalize if normalized
        # ==================================================
        distorted = np.clip(
            distorted,
            0,
            1
        )

        distorted = (
            distorted * 255
        ).astype("uint8")

        # ==================================================
        # FFT
        # ==================================================
        fft = get_fft(distorted)

        # ==================================================
        # FFT Tensor Transform
        # ==================================================
        fft = self.fft_tensor_transform(fft)

        return (
            spatial,
            fft,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )