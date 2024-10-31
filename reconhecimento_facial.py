from google.colab import drive
drive.mount('/content/drive')

!pip install tensorflow matplotlib ultralytics opencv-python-headless

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.image as mpimg

data_dir = '/content/drive/MyDrive/Colab Notebooks/dataset'
IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE = 180, 180, 32

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, epochs=10)

plt.plot(history.history['accuracy'], label='Precisão de Treinamento')
plt.title('Precisão de Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Precisão')
plt.legend()
plt.show()

model.save('walter_jesse_model.h5')

yolo_model = YOLO('yolov8n.pt')
classifier_model = tf.keras.models.load_model('walter_jesse_model.h5')

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (IMG_HEIGHT, IMG_WIDTH))
    face_img = np.expand_dims(face_img, axis=0) / 255.0
    return face_img

test_image_path = '/content/drive/MyDrive/Colab Notebooks/teste3.jpg'
results = yolo_model(test_image_path)
image = cv2.imread(test_image_path)
confidence_threshold = 0.5

for box in results[0].boxes:
    if box.conf[0] < confidence_threshold:
        continue
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    face = image[y1:y2, x1:x2]
    preprocessed_face = preprocess_face(face)
    predictions = classifier_model.predict(preprocessed_face)
    label = "Walter White" if predictions[0] > 0.5 else "Jesse Pinkman"
    color = (0, 255, 0) if label == "Walter White" else (255, 0, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

output_image_path = '/content/image_with_names.jpg'
cv2.imwrite(output_image_path, image)

image_with_names = mpimg.imread(output_image_path)
plt.imshow(image_with_names)
plt.axis('off')
plt.show()
