# Underwater Image Enhancement using GAN

An AI-based underwater image enhancement system using a **Generative Adversarial Network (GAN)** to improve the visual quality of underwater images.

The project uses a **conditional GAN with a U-Net-style Generator and a conditional Discriminator**. The model learns to transform degraded underwater images into enhanced images using paired raw and reference images.

---

## 📌 Project Overview

Underwater images often suffer from:

- Color distortion
- Low contrast
- Poor visibility
- Blur and haze
- Light scattering
- Color imbalance

These problems occur because light behaves differently when it travels through water.

This project uses a deep learning-based GAN approach to enhance underwater images and produce images with improved visual quality.

---

## 🎯 Objectives

The main objectives of this project are:

- Enhance the visual quality of underwater images.
- Reduce underwater color distortion.
- Improve contrast and visibility.
- Restore details in degraded underwater images.
- Use GAN-based learning for image-to-image translation.
- Provide an easy-to-use interface for image enhancement.

---

## 🧠 Model Architecture

The project implements a **conditional GAN / Pix2Pix-style architecture**.

### Generator

The Generator follows a **U-Net-style encoder-decoder architecture**.

```text
Input Underwater Image
        │
        ▼
   Encoder
        │
 ┌──────┴──────┐
 │             │
 ▼             │
Feature        │
Extraction     │
 │             │
 ▼             │
 Bottleneck    │
 │             │
 ▼             │
  Decoder ◄────┘
This allows the Discriminator to determine whether the enhanced image is similar to the expected reference image.

⚙️ Loss Functions

The model uses two major components for Generator training.

1. Adversarial Loss

Binary Cross-Entropy loss is used to encourage the Generator to produce realistic enhanced images.

2. Reconstruction Loss

An L1 loss is calculated between the generated image and the reference image.

The total Generator loss is:

Generator Loss =
Adversarial Loss + λ × Reconstruction Loss

where:

λ = 100

The Discriminator uses Binary Cross-Entropy loss for real and generated image pairs.

🗂️ Project Structure
underwater-image-enhancement-gan/
│
├── models/
│   ├── underwater_enhancement_generator.keras
│   └── underwater_enhancement_discriminator.keras
│
├── src/
│
├── train.py
├── preprocess.py
├── enhance.py
├── evaluate.py
├── compare_results.py
├── test_model.py
├── app.py
│
├── requirements.txt
├── .gitignore
└── README.md
📊 Dataset

The dataset is not included in this GitHub repository because it is large.

Users must download the dataset separately and place it in the project directory.

Dataset Requirements

The training code expects paired images:

Raw Underwater Image
        ↕
Reference Enhanced Image

The raw and reference images must have matching filenames.

For example:

raw/
├── 001.png
├── 002.png
└── 003.png

reference/
├── 001.png
├── 002.png
└── 003.png
Expected Processed Dataset Structure
dataset/
└── processed/
    ├── train/
    │   ├── raw/
    │   └── reference/
    │
    └── val/
        ├── raw/
        └── reference/

Important: The exact dataset download source and preparation procedure should match the dataset used by the project. Download the appropriate paired underwater image dataset and run the preprocessing step before training.

💻 Requirements

The project uses Python and TensorFlow.

Main libraries:

TensorFlow
NumPy
Pillow
OpenCV
Matplotlib
Scikit-learn
Streamlit
🚀 Installation
1. Clone the Repository
git clone https://github.com/avvabika/underwater-image-enhancement-gan.git

Move into the project directory:

cd underwater-image-enhancement-gan
2. Create a Virtual Environment
Windows
python -m venv venv

Activate it:

venv\Scripts\activate
Linux / macOS
python3 -m venv venv

Activate it:

source venv/bin/activate
3. Install Dependencies

Run:

pip install -r requirements.txt
📥 Dataset Preparation

After downloading the dataset, place the required images inside the project dataset directory.

Then run:

python preprocess.py

The preprocessing script prepares the images for training.

The processed data should be available under:

dataset/processed/

with:

dataset/processed/
├── train/
│   ├── raw/
│   └── reference/
│
└── val/
    ├── raw/
    └── reference/
🏋️ Training the GAN

After preparing the dataset, run:

python train.py

The current training configuration is:

Image Size : 256 × 256
Batch Size : 4
Epochs     : 10
Learning Rate : 0.0002
Lambda     : 100

During training, the program:

Loads paired images.
Builds the Generator.
Builds the Discriminator.
Trains both networks.
Calculates Generator and Discriminator losses.
Calculates reconstruction loss.
Generates sample enhanced images.
Saves Generator checkpoints.
🤖 Trained Models

The repository contains trained model files:

models/
├── underwater_enhancement_generator.keras
└── underwater_enhancement_discriminator.keras

The Generator is used for producing enhanced underwater images.

The Discriminator is used during GAN training to distinguish generated images from reference images.

🖼️ Enhancing an Image

After obtaining the trained Generator, use:

python enhance.py

The enhancement script uses the trained Generator to transform an underwater image into an enhanced image.

📈 Evaluation

The project includes an evaluation script:

python evaluate.py

This can be used to evaluate the enhancement results using the evaluation functionality implemented in the project.

🔬 Comparing Results

The project also contains:

python compare_results.py

This script can be used to compare enhancement results.

🧪 Testing the Model

To test the trained model:

python test_model.py
🌐 Streamlit Application

The project includes a Streamlit interface through:

app.py

To launch the application:

streamlit run app.py

After starting the application, Streamlit will provide a local address where the interface can be opened in a web browser.

The application allows users to interact with the underwater image enhancement system without directly working with the Python scripts.

📁 Outputs

Training and enhancement results are stored locally in the outputs/ directory.

Example:

outputs/
├── epoch_001.png
├── epoch_002.png
├── epoch_003.png
└── ...

The output directory is excluded from GitHub to keep the repository lightweight.

🔄 Complete Workflow
        Dataset
           │
           ▼
    preprocess.py
           │
           ▼
   Processed Dataset
           │
           ▼
       train.py
           │
     ┌─────┴─────┐
     ▼           ▼
 Generator   Discriminator
     │           │
     └─────┬─────┘
           │
           ▼
     Trained Model
           │
           ▼
      enhance.py
           │
           ▼
   Enhanced Image
           │
           ▼
      Streamlit
       app.py
🛠️ Technologies Used
Python
TensorFlow
Keras
NumPy
Pillow
OpenCV
Matplotlib
Scikit-learn
Streamlit
Generative Adversarial Networks (GAN)
U-Net architecture
Deep Learning
📌 Current Configuration
Parameter	Value
Image Size	256 × 256
Batch Size	4
Epochs	10
Learning Rate	0.0002
Optimizer	Adam
Beta 1	0.5
Reconstruction Loss	L1
Reconstruction Weight	100
Generator Output	RGB image
Output Activation	Tanh
🔮 Future Enhancements

Possible future improvements include:

Training with larger datasets.
Increasing the number of training epochs.
Adding perceptual loss.
Adding SSIM and PSNR evaluation.
Improving color restoration.
Using attention mechanisms.
Experimenting with different GAN architectures.
Improving inference speed.
Deploying the application as a web service.
Adding side-by-side before/after visualization.
⚠️ Important Notes

The dataset is intentionally excluded from this repository because of its large size.

The venv/, dataset files, ZIP archives, and generated outputs are also excluded through .gitignore.

For training from scratch, users must obtain the required dataset separately and arrange it according to the structure described above.

👩‍💻 Author

Avvabi K A

MCA Student | AI & Machine Learning Enthusiast

GitHub:

https://github.com/avvabika

📜 License

This project is intended for academic and educational purposes.

Please check the license and usage conditions of the dataset separately before redistributing or using it.


### One important thing

I deliberately **didn't put a made-up dataset URL** in the README. Your code requires a particular paired raw/reference arrangement, and we should identify the exact dataset you used before publishing a download link.

Also, your README currently has an empty file in GitHub because we committed it before writing this. Once you paste this into `README.md`, we'll commit and push the README as the next update.
        │
        ▼
Enhanced Image
