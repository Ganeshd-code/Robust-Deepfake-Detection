import pandas as pd
import cv2
import torch

from torch.utils.data import Dataset

from utils.fft import get_fft


class HybridDataset(Dataset):

    def __init__(
        self,
        csv,
        transform_rgb=None,
        transform_fft=None
    ):

        self.data = pd.read_csv(csv)

        self.t_rgb = transform_rgb
        self.t_fft = transform_fft

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
        # ==================================================
        if self.t_rgb:

            rgb = self.t_rgb(img)

        else:

            rgb = img

        # ==================================================
        # Frequency Branch
        # ==================================================
        fft_img = get_fft(img)

        if self.t_fft:

            fft = self.t_fft(fft_img)

        else:

            fft = fft_img

        return (
            rgb,
            fft,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )