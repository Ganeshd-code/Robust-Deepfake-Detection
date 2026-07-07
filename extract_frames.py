import os
import cv2


# ==================================================
# Dataset Video Paths
# ==================================================
REAL_PATH = "FaceForensics++_C23/original"

FAKE_PATHS = [
    "FaceForensics++_C23/Deepfakes",
    "FaceForensics++_C23/FaceSwap"
]


# ==================================================
# Save Paths
# ==================================================
SAVE_REAL = "data/frames/real"
SAVE_FAKE = "data/frames/fake"

os.makedirs(SAVE_REAL, exist_ok=True)
os.makedirs(SAVE_FAKE, exist_ok=True)


# ==================================================
# Frame Extraction Function
# ==================================================
def extract_frames(
    video_path,
    save_folder,
    label,
    max_frames=10,
    frame_skip=5
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print(f"Could not open video: {video_path}")

        return

    frame_count = 0
    saved_count = 0

    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # ------------------------
        # Skip frames
        # ------------------------
        if frame_count % frame_skip == 0:

            filename = (
                f"{label}_"
                f"{video_name}_"
                f"{saved_count}.jpg"
            )

            save_path = os.path.join(
                save_folder,
                filename
            )

            cv2.imwrite(
                save_path,
                frame
            )

            saved_count += 1

        frame_count += 1

        # ------------------------
        # Limit frames
        # ------------------------
        if saved_count >= max_frames:
            break

    cap.release()


# ==================================================
# Process Folder
# ==================================================
def process_folder(
    folder_path,
    save_folder,
    label,
    limit=200
):

    if not os.path.exists(folder_path):

        print(f"Folder not found: {folder_path}")

        return

    videos = os.listdir(folder_path)

    print(
        f"\nProcessing {label.upper()} videos "
        f"from: {folder_path}"
    )

    processed = 0

    for video in videos:

        if processed >= limit:
            break

        video_path = os.path.join(
            folder_path,
            video
        )

        # ------------------------
        # Skip non-video files
        # ------------------------
        if not os.path.isfile(video_path):
            continue

        if not video.lower().endswith((
            ".mp4",
            ".avi",
            ".mov",
            ".mkv"
        )):
            continue

        extract_frames(
            video_path,
            save_folder,
            label
        )

        processed += 1

        if processed % 20 == 0:

            print(
                f"Processed {processed} videos..."
            )

    print(
        f"Finished processing "
        f"{processed} videos."
    )


# ==================================================
# Run Extraction
# ==================================================

# ------------------------
# REAL Videos
# ------------------------
process_folder(
    REAL_PATH,
    SAVE_REAL,
    "real"
)

# ------------------------
# FAKE Videos
# ------------------------
for fake_path in FAKE_PATHS:

    process_folder(
        fake_path,
        SAVE_FAKE,
        "fake"
    )


# ==================================================
# Done
# ==================================================
print("\nFrame extraction completed successfully.")