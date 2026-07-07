import os
import pandas as pd

from sklearn.model_selection import train_test_split


# ==================================================
# Paths
# ==================================================
REAL_PATH = "data/faces/real"
FAKE_PATH = "data/faces/fake"


# ==================================================
# Load Dataset
# ==================================================
def load_data():

    data = []

    # ------------------------
    # REAL IMAGES
    # label = 0
    # ------------------------
    for img in os.listdir(REAL_PATH):

        path = os.path.join(
            REAL_PATH,
            img
        )

        if os.path.isfile(path):

            data.append([
                path,
                0
            ])

    # ------------------------
    # FAKE IMAGES
    # label = 1
    # ------------------------
    for img in os.listdir(FAKE_PATH):

        path = os.path.join(
            FAKE_PATH,
            img
        )

        if os.path.isfile(path):

            data.append([
                path,
                1
            ])

    df = pd.DataFrame(
        data,
        columns=[
            "path",
            "label"
        ]
    )

    return df


# ==================================================
# Create DataFrame
# ==================================================
df = load_data()

print(f"Total Images : {len(df)}")

print(
    "\nClass Distribution:\n",
    df["label"].value_counts()
)


# ==================================================
# Split Dataset
# 70% Train
# 15% Validation
# 15% Test
# ==================================================
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)


# ==================================================
# Create Split Folder
# ==================================================
os.makedirs(
    "data/splits",
    exist_ok=True
)


# ==================================================
# Save CSV Files
# ==================================================
train_df.to_csv(
    "data/splits/train.csv",
    index=False
)

val_df.to_csv(
    "data/splits/val.csv",
    index=False
)

test_df.to_csv(
    "data/splits/test.csv",
    index=False
)


# ==================================================
# Stats
# ==================================================
print("\nDataset Splits Created Successfully.\n")

print(f"Train Size : {len(train_df)}")
print(f"Validation Size : {len(val_df)}")
print(f"Test Size : {len(test_df)}")

print("\nLabel Mapping:")
print("0 = REAL")
print("1 = FAKE")