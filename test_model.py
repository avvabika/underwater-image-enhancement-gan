import tensorflow as tf
from PIL import Image
import numpy as np
import os

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/underwater_enhancement_generator.keras"
INPUT_IMAGE = "dataset/raw/raw-890/100_img_.png"
OUTPUT_IMAGE = "outputs/test_enhanced.png"

print("=" * 60)
print("UNDERWATER IMAGE ENHANCEMENT - MODEL TEST")
print("=" * 60)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Generator...")

generator = tf.keras.models.load_model(MODEL_PATH)

print("Generator loaded successfully ✅")
print("Input shape :", generator.input_shape)
print("Output shape:", generator.output_shape)

# ============================================================
# LOAD IMAGE
# ============================================================

print("\nLoading test image...")

image = Image.open(INPUT_IMAGE).convert("RGB")

print("Original image size:", image.size)

# ============================================================
# PREPROCESS
# ============================================================

image = image.resize((256, 256))

image_array = np.array(image).astype(np.float32)

# Convert [0,255] → [-1,1]
image_array = (image_array / 127.5) - 1.0

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

print("Input tensor shape:", image_array.shape)
print(
    "Input value range:",
    image_array.min(),
    "to",
    image_array.max()
)

# ============================================================
# GENERATE ENHANCED IMAGE
# ============================================================

print("\nGenerating enhanced image...")

enhanced = generator(
    image_array,
    training=False
)

# ============================================================
# POSTPROCESS
# ============================================================

enhanced = enhanced[0].numpy()

# Convert [-1,1] → [0,255]
enhanced = (enhanced + 1.0) * 127.5

enhanced = np.clip(
    enhanced,
    0,
    255
).astype(np.uint8)

# ============================================================
# SAVE RESULT
# ============================================================

os.makedirs("outputs", exist_ok=True)

Image.fromarray(enhanced).save(
    OUTPUT_IMAGE
)

print("\n" + "=" * 60)
print("ENHANCEMENT COMPLETE ✅")
print("=" * 60)

print("Output saved to:")
print(OUTPUT_IMAGE)