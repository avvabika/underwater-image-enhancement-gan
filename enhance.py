import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/underwater_enhancement_generator.keras"

CHALLENGING_DIR = "dataset/challenging/challenging-60"

OUTPUT_DIR = "outputs/challenging_results"

IMAGE_SIZE = (256, 256)

# ============================================================
# START
# ============================================================

print("=" * 60)
print("UNDERWATER IMAGE ENHANCEMENT")
print("CHALLENGING DATASET TEST")
print("=" * 60)

# ============================================================
# CHECK MODEL
# ============================================================

print("\nLoading trained Generator...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Generator model not found:\n{MODEL_PATH}"
    )

generator = tf.keras.models.load_model(MODEL_PATH)

print("Generator loaded successfully ✅")
print("Input shape :", generator.input_shape)
print("Output shape:", generator.output_shape)

# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(CHALLENGING_DIR):
    raise FileNotFoundError(
        f"Challenging dataset not found:\n{CHALLENGING_DIR}"
    )

image_files = sorted(
    glob.glob(os.path.join(CHALLENGING_DIR, "*.png"))
    + glob.glob(os.path.join(CHALLENGING_DIR, "*.jpg"))
    + glob.glob(os.path.join(CHALLENGING_DIR, "*.jpeg"))
)

print("\nChallenging images found:", len(image_files))

if len(image_files) == 0:
    raise RuntimeError("No images found in challenging dataset.")

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Output directory:")
print(OUTPUT_DIR)

# ============================================================
# PROCESS IMAGES
# ============================================================

print("\n" + "=" * 60)
print("STARTING ENHANCEMENT")
print("=" * 60)

for index, image_path in enumerate(image_files, start=1):

    filename = os.path.basename(image_path)

    print(
        f"[{index:02d}/{len(image_files)}] "
        f"Processing: {filename}"
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(image_path).convert("RGB")

    # Keep original size for final output
    original_size = image.size

    # Resize for model
    image_resized = image.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # Convert image to NumPy
    # --------------------------------------------------------

    image_array = np.array(
        image_resized
    ).astype(np.float32)

    # --------------------------------------------------------
    # Normalize [0,255] -> [-1,1]
    # --------------------------------------------------------

    image_array = (
        image_array / 127.5
    ) - 1.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------------
    # Generate enhanced image
    # --------------------------------------------------------

    enhanced = generator(
        image_array,
        training=False
    )

    enhanced = enhanced[0].numpy()

    # --------------------------------------------------------
    # Convert [-1,1] -> [0,255]
    # --------------------------------------------------------

    enhanced = (
        enhanced + 1.0
    ) * 127.5

    enhanced = np.clip(
        enhanced,
        0,
        255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Convert to PIL
    # --------------------------------------------------------

    enhanced_image = Image.fromarray(
        enhanced
    )

    # Restore original image dimensions
    enhanced_image = enhanced_image.resize(
        original_size,
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    enhanced_image.save(
        output_path
    )

print("\n" + "=" * 60)
print("CHALLENGING TEST COMPLETE ✅")
print("=" * 60)

print("Images processed:", len(image_files))
print("Results saved to:")
print(OUTPUT_DIR)