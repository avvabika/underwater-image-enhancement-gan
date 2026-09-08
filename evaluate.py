import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/underwater_enhancement_generator.keras"

RAW_DIR = "dataset/raw/raw-890"
REFERENCE_DIR = "dataset/reference/reference-890"

IMAGE_SIZE = (256, 256)

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

print("\nLoading Generator...")

generator = tf.keras.models.load_model(MODEL_PATH)

print("Generator loaded successfully ✅")

# ============================================================
# FIND MATCHING VALIDATION/REFERENCE IMAGES
# ============================================================

raw_files = {
    os.path.basename(f): f
    for f in glob.glob(os.path.join(RAW_DIR, "*"))
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
}

reference_files = {
    os.path.basename(f): f
    for f in glob.glob(os.path.join(REFERENCE_DIR, "*"))
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
}

common_files = sorted(
    set(raw_files.keys()) & set(reference_files.keys())
)

print("\nRaw images       :", len(raw_files))
print("Reference images:", len(reference_files))
print("Matching images :", len(common_files))

# ============================================================
# USE SAME VALIDATION SPLIT
# ============================================================

# Your validation set contained 178 images.
# We use the last 178 matching files here only as an evaluation
# sample. If your original split used a saved file list, that
# list should be used instead.

evaluation_files = common_files[-178:]

print("Evaluation images:", len(evaluation_files))

# ============================================================
# CALCULATE PSNR AND SSIM
# ============================================================

psnr_values = []
ssim_values = []

print("\nEvaluating images...")

for index, filename in enumerate(evaluation_files, start=1):

    raw = Image.open(
        raw_files[filename]
    ).convert("RGB")

    reference = Image.open(
        reference_files[filename]
    ).convert("RGB")

    raw = raw.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    reference = reference.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    raw_array = np.array(raw).astype(np.float32)
    reference_array = np.array(reference).astype(np.float32)

    # Normalize raw image
    model_input = (
        raw_array / 127.5
    ) - 1.0

    model_input = np.expand_dims(
        model_input,
        axis=0
    )

    # Generate enhanced image
    enhanced = generator(
        model_input,
        training=False
    )[0].numpy()

    # Convert [-1,1] -> [0,255]
    enhanced = (
        enhanced + 1.0
    ) * 127.5

    enhanced = np.clip(
        enhanced,
        0,
        255
    ).astype(np.uint8)

    # PSNR
    psnr = peak_signal_noise_ratio(
        reference_array,
        enhanced,
        data_range=255
    )

    # SSIM
    ssim = structural_similarity(
        reference_array,
        enhanced,
        channel_axis=2,
        data_range=255
    )

    psnr_values.append(psnr)
    ssim_values.append(ssim)

    if index % 20 == 0:
        print(
            f"Processed {index}/{len(evaluation_files)}"
        )

# ============================================================
# FINAL RESULTS
# ============================================================

average_psnr = np.mean(psnr_values)
average_ssim = np.mean(ssim_values)

print("\n" + "=" * 60)
print("FINAL EVALUATION RESULTS")
print("=" * 60)

print(
    f"Number of images : {len(evaluation_files)}"
)

print(
    f"Average PSNR     : {average_psnr:.4f} dB"
)

print(
    f"Average SSIM     : {average_ssim:.4f}"
)

print("=" * 60)