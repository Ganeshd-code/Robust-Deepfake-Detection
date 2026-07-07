from torchvision import transforms


# ==================================================
# TRAINING TRANSFORMS
# Used ONLY during training
# ==================================================

# ------------------------
# Spatial Branch Training
# Mild augmentation
# ------------------------
def get_spatial_train_transform():

    return transforms.Compose([

        # numpy -> PIL
        transforms.ToPILImage(),

        # fixed size
        transforms.Resize((224, 224)),

        # mild blur
        transforms.RandomApply([
            transforms.GaussianBlur(
                kernel_size=3
            )
        ], p=0.2),

        # mild color variation
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            )
        ], p=0.2),

        # tensor conversion
        transforms.ToTensor(),

        # normalization
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ------------------------
# Frequency Branch Training
# stronger augmentation
# ------------------------
def get_frequency_train_transform():

    return transforms.Compose([

        transforms.ToPILImage(),

        transforms.Resize((224, 224)),

        # stronger blur
        transforms.RandomApply([
            transforms.GaussianBlur(
                kernel_size=5
            )
        ], p=0.3),

        # stronger distortion
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            )
        ], p=0.3),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ==================================================
# INFERENCE / VALIDATION TRANSFORMS
# Deterministic (NO randomness)
# ==================================================

# ------------------------
# Spatial Inference
# ------------------------
def get_spatial_infer_transform():

    return transforms.Compose([

        transforms.ToPILImage(),

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ------------------------
# Frequency Inference
# ------------------------
def get_frequency_infer_transform():

    return transforms.Compose([

        transforms.ToPILImage(),

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])