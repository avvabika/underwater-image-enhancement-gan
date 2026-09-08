import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = 256
BATCH_SIZE = 4

# Start small to verify training works.
# Later we can increase this.
EPOCHS = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_RAW = os.path.join(
    BASE_DIR, "dataset", "processed", "train", "raw"
)

TRAIN_REF = os.path.join(
    BASE_DIR, "dataset", "processed", "train", "reference"
)

VAL_RAW = os.path.join(
    BASE_DIR, "dataset", "processed", "val", "raw"
)

VAL_REF = os.path.join(
    BASE_DIR, "dataset", "processed", "val", "reference"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# GPU CHECK
# ============================================================

print("=" * 60)
print("DEVICE CHECK")
print("=" * 60)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detected:")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU detected.")
    print("Training will use CPU.")

# ============================================================
# LOAD IMAGE PAIRS
# ============================================================

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype(np.float32)

    # Convert [0,255] -> [-1,1]
    image = (image / 127.5) - 1.0

    return image


def load_pairs(raw_dir, ref_dir):
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.png")))

    raw_images = []
    ref_images = []

    for raw_path in raw_files:
        filename = os.path.basename(raw_path)
        ref_path = os.path.join(ref_dir, filename)

        if not os.path.exists(ref_path):
            print("WARNING: Missing reference:", filename)
            continue

        raw_images.append(load_image(raw_path))
        ref_images.append(load_image(ref_path))

    return np.array(raw_images), np.array(ref_images)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING TRAINING DATA")
print("=" * 60)

train_raw, train_ref = load_pairs(TRAIN_RAW, TRAIN_REF)

print("Train Raw       :", train_raw.shape)
print("Train Reference :", train_ref.shape)

# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING VALIDATION DATA")
print("=" * 60)

val_raw, val_ref = load_pairs(VAL_RAW, VAL_REF)

print("Val Raw       :", val_raw.shape)
print("Val Reference :", val_ref.shape)

# ============================================================
# TF DATASETS
# ============================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_raw, train_ref)
)

train_dataset = (
    train_dataset
    .shuffle(len(train_raw))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_raw, val_ref)
)

val_dataset = (
    val_dataset
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ============================================================
# GENERATOR
# ============================================================

def build_generator():

    inputs = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3),
        name="underwater_input"
    )

    # -------------------------
    # Encoder
    # -------------------------

    e1 = layers.Conv2D(
        64, 4, strides=2, padding="same"
    )(inputs)

    e1 = layers.LeakyReLU(0.2)(e1)

    e2 = layers.Conv2D(
        128, 4, strides=2, padding="same"
    )(e1)

    e2 = layers.BatchNormalization()(e2)
    e2 = layers.LeakyReLU(0.2)(e2)

    e3 = layers.Conv2D(
        256, 4, strides=2, padding="same"
    )(e2)

    e3 = layers.BatchNormalization()(e3)
    e3 = layers.LeakyReLU(0.2)(e3)

    e4 = layers.Conv2D(
        512, 4, strides=2, padding="same"
    )(e3)

    e4 = layers.BatchNormalization()(e4)
    e4 = layers.LeakyReLU(0.2)(e4)

    e5 = layers.Conv2D(
        512, 4, strides=2, padding="same"
    )(e4)

    e5 = layers.LeakyReLU(0.2)(e5)

    # -------------------------
    # Decoder
    # -------------------------

    d4 = layers.Conv2DTranspose(
        512, 4, strides=2, padding="same"
    )(e5)

    d4 = layers.BatchNormalization()(d4)
    d4 = layers.ReLU()(d4)

    d4 = layers.Concatenate()([d4, e4])

    d3 = layers.Conv2DTranspose(
        256, 4, strides=2, padding="same"
    )(d4)

    d3 = layers.BatchNormalization()(d3)
    d3 = layers.ReLU()(d3)

    d3 = layers.Concatenate()([d3, e3])

    d2 = layers.Conv2DTranspose(
        128, 4, strides=2, padding="same"
    )(d3)

    d2 = layers.BatchNormalization()(d2)
    d2 = layers.ReLU()(d2)

    d2 = layers.Concatenate()([d2, e2])

    d1 = layers.Conv2DTranspose(
        64, 4, strides=2, padding="same"
    )(d2)

    d1 = layers.BatchNormalization()(d1)
    d1 = layers.ReLU()(d1)

    d1 = layers.Concatenate()([d1, e1])

    outputs = layers.Conv2DTranspose(
        3,
        4,
        strides=2,
        padding="same",
        activation="tanh",
        name="enhanced_output"
    )(d1)

    return Model(inputs, outputs, name="Generator")


# ============================================================
# DISCRIMINATOR
# ============================================================

def build_discriminator():

    raw_input = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3),
        name="raw_image"
    )

    enhanced_input = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3),
        name="enhanced_image"
    )

    x = layers.Concatenate()([
        raw_input,
        enhanced_input
    ])

    x = layers.Conv2D(
        64, 4, strides=2, padding="same"
    )(x)

    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(
        128, 4, strides=2, padding="same"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(
        256, 4, strides=2, padding="same"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(
        512, 4, strides=1, padding="same"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    outputs = layers.Conv2D(
        1, 4, strides=1, padding="same"
    )(x)

    return Model(
        [raw_input, enhanced_input],
        outputs,
        name="Discriminator"
    )


# ============================================================
# BUILD MODELS
# ============================================================

print("\n" + "=" * 60)
print("BUILDING MODELS")
print("=" * 60)

generator = build_generator()
discriminator = build_discriminator()

print("\nGenerator parameters:",
      generator.count_params())

print("Discriminator parameters:",
      discriminator.count_params())

# ============================================================
# OPTIMIZERS
# ============================================================

generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002,
    beta_1=0.5
)

discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002,
    beta_1=0.5
)

# ============================================================
# LOSSES
# ============================================================

bce = tf.keras.losses.BinaryCrossentropy(
    from_logits=True
)


def discriminator_loss(real_output, fake_output):

    real_loss = bce(
        tf.ones_like(real_output),
        real_output
    )

    fake_loss = bce(
        tf.zeros_like(fake_output),
        fake_output
    )

    return real_loss + fake_loss


def generator_adversarial_loss(fake_output):

    return bce(
        tf.ones_like(fake_output),
        fake_output
    )


def reconstruction_loss(fake_image, target_image):

    return tf.reduce_mean(
        tf.abs(target_image - fake_image)
    )


# Weight of reconstruction loss
LAMBDA = 100.0

# ============================================================
# TRAINING STEP
# ============================================================

@tf.function
def train_step(raw_image, reference_image):

    with tf.GradientTape() as gen_tape, \
         tf.GradientTape() as disc_tape:

        generated_image = generator(
            raw_image,
            training=True
        )

        real_output = discriminator(
            [raw_image, reference_image],
            training=True
        )

        fake_output = discriminator(
            [raw_image, generated_image],
            training=True
        )

        gen_adv_loss = generator_adversarial_loss(
            fake_output
        )

        recon_loss = reconstruction_loss(
            generated_image,
            reference_image
        )

        gen_loss = (
            gen_adv_loss +
            LAMBDA * recon_loss
        )

        disc_loss = discriminator_loss(
            real_output,
            fake_output
        )

    gen_gradients = gen_tape.gradient(
        gen_loss,
        generator.trainable_variables
    )

    disc_gradients = disc_tape.gradient(
        disc_loss,
        discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(
            gen_gradients,
            generator.trainable_variables
        )
    )

    discriminator_optimizer.apply_gradients(
        zip(
            disc_gradients,
            discriminator.trainable_variables
        )
    )

    return gen_loss, disc_loss, recon_loss


# ============================================================
# VALIDATION
# ============================================================

def validation_loss():

    total_loss = 0.0
    count = 0

    for raw_image, reference_image in val_dataset:

        generated_image = generator(
            raw_image,
            training=False
        )

        loss = reconstruction_loss(
            generated_image,
            reference_image
        )

        total_loss += float(loss)
        count += 1

    return total_loss / max(count, 1)


# ============================================================
# SAVE SAMPLE IMAGE
# ============================================================

def save_sample(epoch):

    raw_image = val_raw[0:1]

    generated = generator(
        raw_image,
        training=False
    )

    generated = (generated[0].numpy() + 1.0) * 127.5
    generated = np.clip(generated, 0, 255).astype(np.uint8)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"epoch_{epoch:03d}.png"
    )

    Image.fromarray(generated).save(
        output_path
    )

    print("Sample saved:", output_path)


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "=" * 60)
print("STARTING GAN TRAINING")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):

    print(f"\nEpoch {epoch}/{EPOCHS}")

    gen_losses = []
    disc_losses = []
    recon_losses = []

    for step, (raw_image, reference_image) in enumerate(
        train_dataset
    ):

        gen_loss, disc_loss, recon_loss = train_step(
            raw_image,
            reference_image
        )

        gen_losses.append(float(gen_loss))
        disc_losses.append(float(disc_loss))
        recon_losses.append(float(recon_loss))

        if (step + 1) % 25 == 0:
            print(
                f"Step {step + 1}: "
                f"G={gen_loss:.4f} "
                f"D={disc_loss:.4f} "
                f"Recon={recon_loss:.4f}"
            )

    avg_gen = np.mean(gen_losses)
    avg_disc = np.mean(disc_losses)
    avg_recon = np.mean(recon_losses)

    val_loss = validation_loss()

    print("\n----------------------------------------")
    print(f"Generator Loss      : {avg_gen:.4f}")
    print(f"Discriminator Loss  : {avg_disc:.4f}")
    print(f"Reconstruction Loss : {avg_recon:.4f}")
    print(f"Validation Loss     : {val_loss:.4f}")
    print("----------------------------------------")

    # Save sample
    save_sample(epoch)

    # Save checkpoint
    generator.save(
        os.path.join(
            MODEL_DIR,
            f"generator_epoch_{epoch:03d}.keras"
        )
    )

# ============================================================
# SAVE FINAL MODELS
# ============================================================

print("\n" + "=" * 60)
print("SAVING FINAL MODELS")
print("=" * 60)

generator.save(
    os.path.join(
        MODEL_DIR,
        "generator_final.keras"
    )
)

discriminator.save(
    os.path.join(
        MODEL_DIR,
        "discriminator_final.keras"
    )
)

print("\nGenerator saved:")
print(
    os.path.join(
        MODEL_DIR,
        "generator_final.keras"
    )
)

print("\nDiscriminator saved:")
print(
    os.path.join(
        MODEL_DIR,
        "discriminator_final.keras"
    )
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)