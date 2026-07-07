import cv2
import numpy as np


# ==================================================
# FFT Feature Extraction
# ==================================================
def get_fft(image):

    # ==================================================
    # Safety Check
    # ==================================================
    if image is None:

        raise ValueError(
            "Input image is None."
        )

    # ==================================================
    # Convert to float32
    # improves FFT numerical stability
    # ==================================================
    image = image.astype(np.float32)

    channels = []

    # ==================================================
    # Per-Channel FFT
    # ==================================================
    for i in range(3):

        channel = image[:, :, i]

        # ------------------------
        # FFT
        # ------------------------
        fft = np.fft.fft2(channel)

        # center frequencies
        fft_shift = np.fft.fftshift(fft)

        # ==================================================
        # Magnitude Spectrum
        # ==================================================
        magnitude = np.abs(fft_shift)

        # log scaling
        magnitude = np.log1p(magnitude)

        # ==================================================
        # Normalize
        # ==================================================
        normalized = np.zeros_like(
            magnitude
        )

        magnitude = cv2.normalize(
            magnitude,
            normalized,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX
        )

        magnitude = magnitude.astype(
            np.uint8
        )

        channels.append(magnitude)

    # ==================================================
    # Merge Channels
    # ==================================================
    fft_image = np.stack(
        channels,
        axis=2
    )

    return fft_image