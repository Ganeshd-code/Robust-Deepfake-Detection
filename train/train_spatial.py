import os
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from utils.dataset_spatial import DeepfakeDataset

from utils.transforms import (
    get_spatial_train_transform,
    get_spatial_infer_transform
)

from models.spatial_model import SpatialModel


# ==================================================
# Setup
# ==================================================
os.makedirs("models", exist_ok=True)


# ==================================================
# Dataset
# ==================================================

# ------------------------
# Training Dataset
# ------------------------
train_dataset = DeepfakeDataset(
    "data/splits/train.csv",
    get_spatial_train_transform()
)

# ------------------------
# Validation Dataset
# deterministic inference
# ------------------------
val_dataset = DeepfakeDataset(
    "data/splits/val.csv",
    get_spatial_infer_transform()
)


# ==================================================
# DataLoaders
# ==================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


# ==================================================
# Device
# ==================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==================================================
# Model
# ==================================================
model = SpatialModel().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

criterion = nn.CrossEntropyLoss()


# ==================================================
# Training Setup
# ==================================================
epochs = 5

best_acc = 0.0


# ==================================================
# Epoch Loop
# ==================================================
for epoch in range(epochs):

    # ==================================================
    # TRAINING
    # ==================================================
    model.train()

    total_loss = 0.0

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        # ------------------------
        # Forward
        # ------------------------
        outputs = model(x)

        loss = criterion(outputs, y)

        # ------------------------
        # Backprop
        # ------------------------
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # ==================================================
    # VALIDATION
    # ==================================================
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

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
    acc = accuracy_score(
        all_labels,
        all_preds
    )

    precision = precision_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    try:

        roc_auc = roc_auc_score(
            all_labels,
            all_probs
        )

    except:

        roc_auc = 0.0

    # ==================================================
    # Print Metrics
    # ==================================================
    print(f"""
==================================================
Epoch {epoch+1}/{epochs}

Loss       : {avg_loss:.4f}
Accuracy   : {acc:.4f}
Precision  : {precision:.4f}
Recall     : {recall:.4f}
F1 Score   : {f1:.4f}
ROC-AUC    : {roc_auc:.4f}

==================================================
""")

    # ==================================================
    # Save Best Model
    # ==================================================
    if acc > best_acc:

        best_acc = acc

        torch.save(
            model.state_dict(),
            "models/spatial.pth"
        )

        print(
            f"✔ Saved Best Model "
            f"(Accuracy: {best_acc:.4f})"
        )


# ==================================================
# Training Complete
# ==================================================
print("\nTraining completed successfully.")