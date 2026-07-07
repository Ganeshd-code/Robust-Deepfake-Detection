import os
import cv2


# ==================================================
# Input Folders
# ==================================================
REAL_FRAMES = "data/frames/real"
FAKE_FRAMES = "data/frames/fake"


# ==================================================
# Output Folders
# ==================================================
REAL_FACES = "data/faces/real"
FAKE_FACES = "data/faces/fake"

os.makedirs(REAL_FACES, exist_ok=True)
os.makedirs(FAKE_FACES, exist_ok=True)


# ==================================================
# Face Detector
# ==================================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml" # type: ignore
)

if face_cascade.empty():

    raise RuntimeError(
        "Failed to load Haar Cascade face detector."
    )


# ==================================================
# Face Extraction Function
# ==================================================
def extract_faces(
    input_folder,
    output_folder,
    label
):

    if not os.path.exists(input_folder):

        print(
            f"Input folder not found: "
            f"{input_folder}"
        )

        return

    images = os.listdir(input_folder)

    processed = 0
    saved = 0

    print(
        f"\nExtracting {label.upper()} faces "
        f"from: {input_folder}"
    )

    for img_name in images:

        img_path = os.path.join(
            input_folder,
            img_name
        )

        # ------------------------
        # Skip invalid files
        # ------------------------
        if not os.path.isfile(img_path):
            continue

        if not img_name.lower().endswith((
            ".jpg",
            ".jpeg",
            ".png"
        )):
            continue

        # ------------------------
        # Read Image
        # ------------------------
        img = cv2.imread(img_path)

        if img is None:
            continue

        # ==================================================
        # Resize Large Images
        # improves detection stability
        # ==================================================
        height, width = img.shape[:2]

        max_dim = 1000

        if max(height, width) > max_dim:

            scale = max_dim / max(height, width)

            img = cv2.resize(
                img,
                (
                    int(width * scale),
                    int(height * scale)
                )
            )

        # ==================================================
        # Gray Conversion
        # ==================================================
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        # ==================================================
        # Face Detection
        # ==================================================
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        # ==================================================
        # Skip Images Without Faces
        # ==================================================
        if len(faces) == 0:
            continue

        # ==================================================
        # Keep Largest Face Only
        # ==================================================
        largest_face = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        x, y, w, h = largest_face

        # ==================================================
        # Padding
        # ==================================================
        pad = 20

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)

        x2 = min(
            img.shape[1],
            x + w + pad
        )

        y2 = min(
            img.shape[0],
            y + h + pad
        )

        # ==================================================
        # Crop Face
        # ==================================================
        face = img[y1:y2, x1:x2]

        # ==================================================
        # Resize Face
        # ==================================================
        face = cv2.resize(
            face,
            (224, 224)
        )

        # ==================================================
        # Save
        # ==================================================
        filename = (
            f"{label}_"
            f"{processed}.jpg"
        )

        save_path = os.path.join(
            output_folder,
            filename
        )

        cv2.imwrite(
            save_path,
            face
        )

        saved += 1
        processed += 1

        if processed % 500 == 0:

            print(
                f"Processed {processed} images..."
            )

    print(
        f"Finished. Saved {saved} faces."
    )


# ==================================================
# Run Extraction
# ==================================================

# ------------------------
# REAL
# ------------------------
extract_faces(
    REAL_FRAMES,
    REAL_FACES,
    "real"
)

# ------------------------
# FAKE
# ------------------------
extract_faces(
    FAKE_FRAMES,
    FAKE_FACES,
    "fake"
)


# ==================================================
# Done
# ==================================================
print("\nFace extraction completed successfully.")