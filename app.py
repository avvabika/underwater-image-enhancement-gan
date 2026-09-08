import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io
import time

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/underwater_enhancement_generator.keras"
IMAGE_SIZE = (256, 256)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Underwater Image Enhancement",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(128,128,128,0.25);
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
    }

    .metric-label {
        font-size: 14px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 650;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌊 Underwater Image Enhancement Using GAN</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enhancing underwater images using a Generative Adversarial Network'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🌊 Project")

    st.write(
        """
        This application uses a trained GAN Generator
        to improve the visual quality of underwater images.
        """
    )

    st.divider()

    st.subheader("📊 Dataset")

    st.write("Paired images: **890**")
    st.write("Training images: **712**")
    st.write("Validation images: **178**")
    st.write("Challenging test images: **60**")

    st.divider()

    st.subheader("🧠 Model")

    st.write("Architecture: **GAN**")
    st.write("Input size: **256 × 256**")
    st.write("Optimizer: **Adam**")
    st.write("Learning rate: **0.0002**")
    st.write("Training epochs: **50**")

    st.divider()

    st.subheader("📈 Evaluation")

    st.write("PSNR: **24.9089 dB**")
    st.write("SSIM: **0.8905**")

# ============================================================
# PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">📈 Model Performance</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Training Images",
        "712"
    )

with col2:
    st.metric(
        "Validation Images",
        "178"
    )

with col3:
    st.metric(
        "PSNR",
        "24.9089 dB"
    )

with col4:
    st.metric(
        "SSIM",
        "0.8905"
    )

st.divider()

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_generator():

    if not os.path.exists(MODEL_PATH):
        return None

    return tf.keras.models.load_model(
        MODEL_PATH
    )


generator = load_generator()

if generator is None:

    st.error(
        "❌ Generator model not found.\n\n"
        f"Expected location: `{MODEL_PATH}`"
    )

    st.stop()

# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📤 Upload Underwater Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an underwater image",
    type=["png", "jpg", "jpeg"],
    help="Upload a PNG, JPG or JPEG underwater image."
)

# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.write(
        f"**Original image size:** "
        f"{original_image.width} × {original_image.height}"
    )

    st.divider()

    # --------------------------------------------------------
    # Original preview
    # --------------------------------------------------------

    st.subheader("Original Image")

    st.image(
        original_image,
        use_container_width=True
    )

    st.write("")

    # --------------------------------------------------------
    # Enhance button
    # --------------------------------------------------------

    if st.button(
        "✨ Enhance Image",
        type="primary",
        use_container_width=True
    ):

        start_time = time.time()

        with st.spinner(
            "GAN is enhancing your image..."
        ):

            # Original dimensions
            original_size = original_image.size

            # Resize
            image = original_image.resize(
                IMAGE_SIZE,
                Image.Resampling.LANCZOS
            )

            # NumPy conversion
            image_array = np.array(
                image
            ).astype(np.float32)

            # Normalize [0,255] -> [-1,1]
            image_array = (
                image_array / 127.5
            ) - 1.0

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # GAN inference
            enhanced = generator(
                image_array,
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

            # PIL image
            enhanced_image = Image.fromarray(
                enhanced
            )

            # Restore original dimensions
            enhanced_image = enhanced_image.resize(
                original_size,
                Image.Resampling.LANCZOS
            )

        elapsed = time.time() - start_time

        st.success(
            f"Enhancement completed successfully! ✅ "
            f"Processing time: {elapsed:.2f} seconds"
        )

        st.divider()

        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
            '<div class="section-title">✨ Enhancement Result</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original")

            st.image(
                original_image,
                use_container_width=True
            )

        with col2:

            st.subheader("GAN Enhanced")

            st.image(
                enhanced_image,
                use_container_width=True
            )

        # ====================================================
        # DOWNLOAD
        # ====================================================

        st.divider()

        buffer = io.BytesIO()

        enhanced_image.save(
            buffer,
            format="PNG"
        )

        st.download_button(
            label="⬇️ Download Enhanced Image",
            data=buffer.getvalue(),
            file_name="enhanced_underwater_image.png",
            mime="image/png",
            use_container_width=True
        )

else:

    st.info(
        "👆 Upload an underwater image above to begin."
    )

# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">🔬 About the Project</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    st.subheader("Objective")

    st.write(
        """
        The objective of this project is to enhance underwater
        images affected by poor visibility, color distortion,
        and reduced image quality using a Generative
        Adversarial Network.
        """
    )

with col2:

    st.subheader("Methodology")

    st.write(
        """
        Paired underwater images are used for training.
        The Generator learns to transform degraded underwater
        images into enhanced images while the Discriminator
        helps improve the realism of the generated results.
        """
    )

# ============================================================
# FINAL RESULTS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">🏆 Final Results</div>',
    unsafe_allow_html=True
)

st.write(
    """
    The trained model was evaluated using 178 images.
    The obtained average evaluation metrics were:
    """
)

results_col1, results_col2 = st.columns(2)

with results_col1:

    st.metric(
        "Average PSNR",
        "24.9089 dB"
    )

with results_col2:

    st.metric(
        "Average SSIM",
        "0.8905"
    )

st.write(
    """
    In addition, the trained Generator was tested on a separate
    challenging dataset containing 60 underwater images.
    """
)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Underwater Image Enhancement Using Generative Adversarial Network | MCA Project"
)