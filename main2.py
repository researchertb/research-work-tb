import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define paths
base_path = r'C:/Users/hp/PycharmProjects/pythonProject5'
cxr_path = os.path.join(base_path, 'CXR_png')
clinical_path = os.path.join(base_path, 'ClinicalReadings')
left_mask_path = os.path.join(base_path, 'leftMask')
right_mask_path = os.path.join(base_path, 'rightMask')


# Function to load and preprocess images and masks
def load_image_and_masks(img_path, left_mask_path, right_mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.keras.applications.resnet50.preprocess_input(img)

    left_mask = tf.io.read_file(left_mask_path)
    left_mask = tf.image.decode_png(left_mask, channels=1)
    left_mask = tf.image.resize(left_mask, [224, 224])
    left_mask = tf.cast(left_mask, tf.float32) / 255.0

    right_mask = tf.io.read_file(right_mask_path)
    right_mask = tf.image.decode_png(right_mask, channels=1)
    right_mask = tf.image.resize(right_mask, [224, 224])
    right_mask = tf.cast(right_mask, tf.float32) / 255.0

    combined_mask = tf.maximum(left_mask, right_mask)

    return img, combined_mask


# Function to load clinical data
def load_clinical_data(file_path):
    with open(file_path, 'r') as f:
        data = f.read().strip().split('\n')
    age = int(data[1].split(':')[1].strip()[:-1])
    sex = 1 if data[0].split(':')[1].strip() == 'M' else 0
    return [age, sex]


# Load data
images, masks, clinical_data, labels = [], [], [], []

for filename in os.listdir(cxr_path):
    if filename.endswith('.png'):
        img_path = os.path.join(cxr_path, filename)
        left_mask_file = os.path.join(left_mask_path, filename)
        right_mask_file = os.path.join(right_mask_path, filename)
        clinical_file = os.path.join(clinical_path, filename.replace('.png', '.txt'))

        img, mask = load_image_and_masks(img_path, left_mask_file, right_mask_file)
        images.append(img)
        masks.append(mask)
        clinical_data.append(load_clinical_data(clinical_file))
        labels.append(1 if filename.endswith('_1.png') else 0)

images = np.array(images)
masks = np.array(masks)
clinical_data = np.array(clinical_data)
labels = np.array(labels)

# Split the data
X_train, X_test, m_train, m_test, c_train, c_test, y_train, y_test = train_test_split(
    images, masks, clinical_data, labels, test_size=0.1, random_state=42)

# Normalize clinical data
scaler = StandardScaler()
c_train_scaled = scaler.fit_transform(c_train)
c_test_scaled = scaler.transform(c_test)


# Transformer block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = layers.Add()([x, inputs])

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return layers.Add()([x, res])


# Positional encoding layer
class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=np.arange(position)[:, np.newaxis],
            i=np.arange(d_model)[np.newaxis, :],
            d_model=d_model)

        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


# Vision Transformer model
def create_vit_model(input_shape=(224, 224, 3), patch_size=16, num_classes=1):
    inputs = layers.Input(shape=input_shape)
    patches = layers.Conv2D(256, kernel_size=patch_size, strides=patch_size, padding="valid")(inputs)
    patch_dims = patches.shape[-1]
    patches = layers.Reshape((-1, patch_dims))(patches)

    patch_dim = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)

    pos_embed = PositionalEncoding(patch_dim, patch_dims)(patches)

    embed = pos_embed

    for _ in range(8):
        embed = transformer_encoder(embed, head_size=64, num_heads=4, ff_dim=4 * 256)

    x = layers.GlobalAveragePooling1D()(embed)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    return keras.Model(inputs, outputs)


# Mask generation model
def create_mask_model(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)

    # Transformer middle
    shape_before_flatten = x.shape[1:]
    x = layers.Reshape((-1, shape_before_flatten[-1]))(x)
    x = transformer_encoder(x, head_size=32, num_heads=2, ff_dim=64)
    x = layers.Reshape(shape_before_flatten)(x)

    # Decoder
    x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)

    return models.Model(inputs, outputs)


# Clinical data transformer
def create_clinical_transformer(input_shape=(2,)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(64)(inputs)
    x = layers.Reshape((1, 64))(x)
    x = transformer_encoder(x, head_size=32, num_heads=2, ff_dim=128)
    x = layers.GlobalAveragePooling1D()(x)
    return keras.Model(inputs, x)


# TB prediction model
def create_tb_model(input_shape=(224, 224, 3), mask_shape=(224, 224, 1), clinical_shape=(2,)):
    # Image branch (Vision Transformer)
    img_input = layers.Input(shape=input_shape)
    vit_model = create_vit_model(input_shape)
    x = vit_model(img_input)

    # Mask branch
    mask_input = layers.Input(shape=mask_shape)
    mask_model = create_mask_model(mask_shape)
    m = mask_model(mask_input)
    m = layers.GlobalAveragePooling2D()(m)

    # Clinical data branch
    clinical_input = layers.Input(shape=clinical_shape)
    clinical_transformer = create_clinical_transformer(clinical_shape)
    c = clinical_transformer(clinical_input)

    # Combine branches
    combined = layers.concatenate([x, m, c])
    combined = layers.Dense(256)(combined)
    combined = layers.Reshape((1, 256))(combined)

    # Final transformer for TB prediction
    for _ in range(4):  # 4 transformer layers
        combined = transformer_encoder(combined, head_size=64, num_heads=4, ff_dim=512, dropout=0.1)

    combined = layers.GlobalAveragePooling1D()(combined)
    output = layers.Dense(1, activation='sigmoid')(combined)

    return models.Model(inputs=[img_input, mask_input, clinical_input], outputs=output)


# Create and compile models
mask_model = create_mask_model()
mask_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

tb_model = create_tb_model()
tb_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train mask generation model
mask_history = mask_model.fit(X_train, m_train, validation_data=(X_test, m_test), epochs=20, batch_size=32)

# Generate masks
train_generated_masks = mask_model.predict(X_train)
test_generated_masks = mask_model.predict(X_test)

# Train TB prediction model
tb_history = tb_model.fit(
    [X_train, train_generated_masks, c_train_scaled], y_train,
    validation_data=([X_test, test_generated_masks, c_test_scaled], y_test),
    epochs=50,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
)

# Evaluate the TB prediction model
test_loss, test_accuracy = tb_model.evaluate([X_test, test_generated_masks, c_test_scaled], y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save models
mask_model.save('mask_generation_model.keras')
tb_model.save('tb_prediction_model.keras')


# Prediction function
def predict_tb(image_path, age, sex):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.keras.applications.resnet50.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    generated_mask = mask_model.predict(img)

    clinical = np.array([[age, sex]])
    clinical_scaled = scaler.transform(clinical)

    prediction = tb_model.predict([img, generated_mask, clinical_scaled])[0][0]
    return prediction


# Example usage
sample_image_path = r'C:\Users\hp\PycharmProjects\pythonProject5\CXR_png\MCUCXR_0026_0.png'
sample_age = 10
sample_sex = 1  # 0 for female, 1 for male
prediction = predict_tb(sample_image_path, sample_age, sample_sex)
print(f"Probability of TB: {prediction:.4f}")