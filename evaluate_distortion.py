import torch
import numpy as np

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from torchvision import transforms

from utils.dataset_spatial import DeepfakeDataset
from utils.dataset_freq import DeepfakeFreqDataset
from utils.dataset_hybrid import HybridDataset
from utils.dataset_asymmetric import AsymmetricDataset

from utils.transforms import (
    get_spatial_infer_transform,
    get_frequency_infer_transform
)

from models.spatial_model import SpatialModel
from models.frequency_model import FrequencyModel
from models.hybrid_model import HybridModel


# ==================================================
# Device
# ==================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==================================================
# Distorted Evaluation Transform
# deterministic distortion
# ==================================================
def get_distorted_transform():

    return transforms.Compose([

        transforms.ToPILImage(),

        transforms.Resize((224, 224)),

        transforms.GaussianBlur(
            kernel_size=7
        ),

        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.3
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ==================================================
# Evaluation Function
# ==================================================
def evaluate(
    model,
    loader,
    is_hybrid=False
):

    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():

        for batch in loader:

            # ==================================================
            # Hybrid / Asymmetric
            # ==================================================
            if is_hybrid:

                x1, x2, y = batch

                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)

                outputs = model(x1, x2)

            # ==================================================
            # Spatial / Frequency
            # ==================================================
            else:

                x, y = batch

                x = x.to(device)
                y = y.to(device)

                outputs = model(x)

            # ==================================================
            # Probabilities
            # ==================================================
            probs = torch.softmax(
                outputs,
                dim=1
            )

            preds = torch.argmax(
                probs,
                dim=1
            )

            # ==================================================
            # LABEL MAPPING
            #
            # 0 = REAL
            # 1 = FAKE
            # ==================================================
            fake_probs = probs[:, 1]

            all_labels.extend(
                y.cpu().numpy()
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_probs.extend(
                fake_probs.cpu().numpy()
            )

    # ==================================================
    # Convert Arrays
    # ==================================================
    all_labels = np.array(all_labels)

    all_preds = np.array(all_preds)

    all_probs = np.array(all_probs)

    # ==================================================
    # Metrics
    # ==================================================
    results = {

        "acc": float(
            accuracy_score(
                all_labels,
                all_preds
            )
        ),

        "prec": float(
            precision_score(
                all_labels,
                all_preds,
                zero_division=0
            )
        ),

        "rec": float(
            recall_score(
                all_labels,
                all_preds,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                all_labels,
                all_preds,
                zero_division=0
            )
        )
    }

    # ==================================================
    # ROC-AUC
    # ==================================================
    try:

        results["auc"] = float(
            roc_auc_score(
                all_labels,
                all_probs
            )
        )

    except:

        results["auc"] = 0.0

    return results


# ==================================================
# Load Models
# ==================================================

# ------------------------
# Spatial
# ------------------------
spatial = SpatialModel().to(device)

spatial.load_state_dict(
    torch.load(
        "models/spatial.pth",
        map_location=device,
        weights_only=True
    )
)

spatial.eval()

# ------------------------
# Frequency
# ------------------------
frequency = FrequencyModel().to(device)

frequency.load_state_dict(
    torch.load(
        "models/frequency.pth",
        map_location=device,
        weights_only=True
    )
)

frequency.eval()

# ------------------------
# Hybrid
# ------------------------
hybrid = HybridModel().to(device)

hybrid.load_state_dict(
    torch.load(
        "models/hybrid.pth",
        map_location=device,
        weights_only=True
    )
)

hybrid.eval()

# ------------------------
# Asymmetric
# ------------------------
asymmetric = HybridModel().to(device)

asymmetric.load_state_dict(
    torch.load(
        "models/asymmetric.pth",
        map_location=device,
        weights_only=True
    )
)

asymmetric.eval()


# ==================================================
# CLEAN DATASETS
# ==================================================

# ------------------------
# Spatial
# ------------------------
clean_spatial = DeepfakeDataset(
    "data/splits/val.csv",
    get_spatial_infer_transform()
)

# ------------------------
# Frequency
# ------------------------
clean_frequency = DeepfakeFreqDataset(
    "data/splits/val.csv",
    get_frequency_infer_transform()
)

# ------------------------
# Hybrid
# ------------------------
clean_hybrid = HybridDataset(
    "data/splits/val.csv",

    get_spatial_infer_transform(),

    get_frequency_infer_transform()
)

# ------------------------
# Asymmetric
# ------------------------
clean_asymmetric = AsymmetricDataset(
    "data/splits/val.csv",

    get_spatial_infer_transform(),

    get_frequency_infer_transform()
)


# ==================================================
# DISTORTED DATASETS
# ==================================================

# ------------------------
# Spatial
# ------------------------
distorted_spatial = DeepfakeDataset(
    "data/splits/val.csv",
    get_distorted_transform()
)

# ------------------------
# Frequency
# ------------------------
distorted_frequency = DeepfakeFreqDataset(
    "data/splits/val.csv",
    get_distorted_transform()
)

# ------------------------
# Hybrid
# ------------------------
distorted_hybrid = HybridDataset(
    "data/splits/val.csv",

    get_distorted_transform(),

    get_distorted_transform()
)

# ------------------------
# Asymmetric
# ------------------------
distorted_asymmetric = AsymmetricDataset(
    "data/splits/val.csv",

    get_distorted_transform(),

    get_distorted_transform()
)


# ==================================================
# DataLoader Helper
# ==================================================
def get_loader(dataset):

    return DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )


# ==================================================
# CLEAN EVALUATION
# ==================================================
print("\n================ CLEAN DATA ================\n")

print(
    "Spatial    :",
    evaluate(
        spatial,
        get_loader(clean_spatial)
    )
)

print(
    "Frequency  :",
    evaluate(
        frequency,
        get_loader(clean_frequency)
    )
)

print(
    "Hybrid     :",
    evaluate(
        hybrid,
        get_loader(clean_hybrid),
        is_hybrid=True
    )
)

print(
    "Asymmetric :",
    evaluate(
        asymmetric,
        get_loader(clean_asymmetric),
        is_hybrid=True
    )
)


# ==================================================
# DISTORTED EVALUATION
# ==================================================
print("\n============= DISTORTED DATA =============\n")

print(
    "Spatial    :",
    evaluate(
        spatial,
        get_loader(distorted_spatial)
    )
)

print(
    "Frequency  :",
    evaluate(
        frequency,
        get_loader(distorted_frequency)
    )
)

print(
    "Hybrid     :",
    evaluate(
        hybrid,
        get_loader(distorted_hybrid),
        is_hybrid=True
    )
)

print(
    "Asymmetric :",
    evaluate(
        asymmetric,
        get_loader(distorted_asymmetric),
        is_hybrid=True
    )
)