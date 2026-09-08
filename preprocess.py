import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split

# ==============================
# PATHS
# ==============================

BASE_DIR = r"C:\Users\Avvabi K A\Desktop\gan"

RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw", "raw-890")
REFERENCE_DIR = os.path.join(BASE_DIR, "dataset", "reference", "reference-890")

PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")

# Image size for GAN training
IMAGE_SIZE = (256, 256)

# Validation split
VAL_SIZE = 0.20

# ==============================
# CREATE OUTPUT DIRECTORIES
# ==============================

train_raw = os.path.join(PROCESSED_DIR, "train", "raw")
train_reference = os.path.join(PROCESSED_DIR, "train", "reference")

val_raw = os.path.join(PROCESSED_DIR, "val", "raw")
val_reference = os.path.join(PROCESSED_DIR, "val", "reference")

for folder in [
    train_raw,
    train_reference,
    val_raw,
    val_reference
]:
    os.makedirs(folder, exist_ok=True)

# ==============================
# GET IMAGE FILES
# ==============================

raw_files = {
    f for f in os.listdir(RAW_DIR)
    if f.lower().endswith(".png")
}

reference_files = {
    f for f in os.listdir(REFERENCE_DIR)
    if f.lower().endswith(".png")
}

# ==============================
# VERIFY PAIRS
# ==============================

common_files = sorted(raw_files.intersection(reference_files))

print("=" * 50)
print("DATASET VERIFICATION")
print("=" * 50)

print(f"Raw images       : {len(raw_files)}")
print(f"Reference images : {len(reference_files)}")
print(f"Paired images    : {len(common_files)}")
print(f"Only Raw         : {len(raw_files - reference_files)}")
print(f"Only Reference   : {len(reference_files - raw_files)}")

if len(common_files) == 0:
    raise ValueError("No matching Raw/Reference image pairs found.")

if raw_files != reference_files:
    raise ValueError("Raw and Reference filenames do not match.")

print("\nAll filenames match successfully.")

# ==============================
# TRAIN / VALIDATION SPLIT
# ==============================

train_files, val_files = train_test_split(
    common_files,
    test_size=VAL_SIZE,
    random_state=42,
    shuffle=True
)

print("\n" + "=" * 50)
print("DATASET SPLIT")
print("=" * 50)

print(f"Training images   : {len(train_files)}")
print(f"Validation images : {len(val_files)}")

# ==============================
# PROCESS FUNCTION
# ==============================

def process_image(source_path, destination_path):

    try:
        image = Image.open(source_path).convert("RGB")

        # Resize to 256 × 256
        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS
        )

        # Save as PNG
        image.save(destination_path, format="PNG")

    except Exception as e:
        print(f"ERROR processing {source_path}")
        print(e)


# ==============================
# PROCESS TRAINING DATA
# ==============================

print("\nProcessing training images...")

for filename in train_files:

    raw_source = os.path.join(RAW_DIR, filename)
    reference_source = os.path.join(REFERENCE_DIR, filename)

    raw_destination = os.path.join(train_raw, filename)
    reference_destination = os.path.join(train_reference, filename)

    process_image(raw_source, raw_destination)
    process_image(reference_source, reference_destination)


# ==============================
# PROCESS VALIDATION DATA
# ==============================

print("Processing validation images...")

for filename in val_files:

    raw_source = os.path.join(RAW_DIR, filename)
    reference_source = os.path.join(REFERENCE_DIR, filename)

    raw_destination = os.path.join(val_raw, filename)
    reference_destination = os.path.join(val_reference, filename)

    process_image(raw_source, raw_destination)
    process_image(reference_source, reference_destination)


# ==============================
# FINAL VERIFICATION
# ==============================

train_raw_count = len(os.listdir(train_raw))
train_reference_count = len(os.listdir(train_reference))

val_raw_count = len(os.listdir(val_raw))
val_reference_count = len(os.listdir(val_reference))

print("\n" + "=" * 50)
print("PREPROCESSING COMPLETE")
print("=" * 50)

print(f"Train Raw       : {train_raw_count}")
print(f"Train Reference : {train_reference_count}")
print(f"Val Raw         : {val_raw_count}")
print(f"Val Reference   : {val_reference_count}")

print("\nOutput directory:")
print(PROCESSED_DIR)

print("\nChallenging dataset has NOT been modified.")
print("It will be used later for testing.")