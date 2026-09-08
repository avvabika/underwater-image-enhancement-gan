import os
import math
import glob
import matplotlib.pyplot as plt
from PIL import Image

RAW_DIR = "dataset/challenging/challenging-60"
ENHANCED_DIR = "outputs/challenging_results"

OUTPUT_FILE = "outputs/challenging_comparison.png"

# Get matching image filenames
raw_files = {
    os.path.basename(f)
    for f in glob.glob(os.path.join(RAW_DIR, "*"))
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
}

enhanced_files = {
    os.path.basename(f)
    for f in glob.glob(os.path.join(ENHANCED_DIR, "*"))
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
}

common_files = sorted(raw_files & enhanced_files)

print("=" * 60)
print("CHALLENGING DATASET COMPARISON")
print("=" * 60)

print("Raw images      :", len(raw_files))
print("Enhanced images :", len(enhanced_files))
print("Matching images :", len(common_files))

# Select first 6 images
selected = common_files[:6]

fig, axes = plt.subplots(
    len(selected),
    2,
    figsize=(10, 4 * len(selected))
)

if len(selected) == 1:
    axes = [axes]

for row, filename in enumerate(selected):

    raw_path = os.path.join(
        RAW_DIR,
        filename
    )

    enhanced_path = os.path.join(
        ENHANCED_DIR,
        filename
    )

    raw = Image.open(raw_path).convert("RGB")
    enhanced = Image.open(enhanced_path).convert("RGB")

    axes[row][0].imshow(raw)
    axes[row][0].set_title(
        f"Raw - {filename}"
    )
    axes[row][0].axis("off")

    axes[row][1].imshow(enhanced)
    axes[row][1].set_title(
        f"GAN Enhanced - {filename}"
    )
    axes[row][1].axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\nComparison saved successfully ✅")
print("Output:", OUTPUT_FILE)