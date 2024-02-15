import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers

number_2_expression = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6,
}


def load_data(folder_path):
    images = []
    labels = []

    for label_name in os.listdir(folder_path):
        label_folder_path = os.path.join(folder_path, label_name)
        for image_name in os.listdir(label_folder_path):
            image_path = os.path.join(label_folder_path, image_name)
            image = Image.open(image_path).convert('RGB')
            image_nd_array = np.array(image) / 255.0

            labels.append(number_2_expression[label_name])
            images.append(image_nd_array.flatten())

    nd_array_images = np.array(images)
    nd_array_labels = np.array(labels)

    return train_test_split(
        nd_array_images, nd_array_labels, test_size=0.2, random_state=42)


def train(images, labels, save_path):
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    model.fit(images, labels, epochs=10)
    model.save(save_path)
    return model
