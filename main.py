import os
import numpy as np
import tensorflow as tf
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
    # Load and preprocess main image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.keras.applications.resnet50.preprocess_input(img)

    # Load and preprocess left mask
    left_mask = tf.io.read_file(left_mask_path)
    left_mask = tf.image.decode_png(left_mask, channels=1)
    left_mask = tf.image.resize(left_mask, [224, 224])
    left_mask = tf.cast(left_mask, tf.float32) / 255.0

    # Load and preprocess right mask
    right_mask = tf.io.read_file(right_mask_path)
    right_mask = tf.image.decode_png(right_mask, channels=1)
    right_mask = tf.image.resize(right_mask, [224, 224])
    right_mask = tf.cast(right_mask, tf.float32) / 255.0

    # Combine left and right masks
    combined_mask = tf.maximum(left_mask, right_mask)

    return img, combined_mask


# Function to load clinical data
def load_clinical_data(file_path):
    with open(file_path, 'r') as f:
        data = f.read().strip().split('\n')
    age = int(data[1].split(':')[1].strip()[:-1])  # Extract age
    sex = 1 if data[0].split(':')[1].strip() == 'M' else 0  # Convert sex to binary
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


# Define the mask generation model
def create_mask_model(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)

    # Decoder
    x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)

    return models.Model(inputs, outputs)


# Define the TB prediction model
def create_tb_model(input_shape=(224, 224, 3), mask_shape=(224, 224, 1), clinical_shape=(2,)):
    # Image branch
    img_input = layers.Input(shape=input_shape)
    x = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')(img_input)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)

    # Mask branch
    mask_input = layers.Input(shape=mask_shape)
    m = layers.Conv2D(32, 3, activation='relu')(mask_input)
    m = layers.MaxPooling2D()(m)
    m = layers.Flatten()(m)
    m = layers.Dense(64, activation='relu')(m)

    # Clinical data branch
    clinical_input = layers.Input(shape=clinical_shape)
    c = layers.Dense(32, activation='relu')(clinical_input)

    # Combine branches
    combined = layers.concatenate([x, m, c])
    combined = layers.Dense(64, activation='relu')(combined)
    output = layers.Dense(1, activation='sigmoid')(combined)

    return models.Model(inputs=[img_input, mask_input, clinical_input], outputs=output)


# Create and compile the mask generation model
mask_model = create_mask_model()
mask_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the mask generation model
mask_history = mask_model.fit(X_train, m_train, validation_data=(X_test, m_test), epochs=20, batch_size=32)

# Generate masks for all images
train_generated_masks = mask_model.predict(X_train)
test_generated_masks = mask_model.predict(X_test)

# Create and compile the TB prediction model
tb_model = create_tb_model()
tb_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the TB prediction model
tb_history = tb_model.fit(
    [X_train, train_generated_masks, c_train_scaled], y_train,
    validation_data=([X_test, test_generated_masks, c_test_scaled], y_test),
    epochs=10,
    batch_size=32
)

# Evaluate the TB prediction model
test_loss, test_accuracy = tb_model.evaluate([X_test, test_generated_masks, c_test_scaled], y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save the models
mask_model.save('mask_generation_model.keras')
tb_model.save('tb_prediction_model.keras')


# Function to make predictions
def predict_tb(image_path, age, sex):
    # Load and preprocess image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.keras.applications.resnet50.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Generate mask
    generated_mask = mask_model.predict(img)

    # Prepare clinical data
    clinical = np.array([[age, sex]])
    clinical_scaled = scaler.transform(clinical)

    # Predict TB
    prediction = tb_model.predict([img, generated_mask, clinical_scaled])[0][0]
    return prediction


# Example usage
sample_image_path = r'C:\Users\hp\PycharmProjects\pythonProject5\CXR_png\MCUCXR_0026_0.png'
sample_age = 10
sample_sex = 1  # 0 for female, 1 for male
prediction = predict_tb(sample_image_path, sample_age, sample_sex)
print(f"Probability of TB: {prediction:.4f}")
