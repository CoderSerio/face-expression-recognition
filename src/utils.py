import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

expression_2_number = ['angry', 'disgust',
                       'fear', 'happy', 'neutral', 'sad', 'surprise']
number_2_expression = {expression: index for index,
                       expression in enumerate(expression_2_number)}


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
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))

    optimizer = Adam(learning_rate=0.00005)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    model.fit(images, labels, epochs=30)
    model.save(save_path)
    return model


def predict(model, nd_array_images, labels):
    prediction = model.predict(nd_array_images)
    # 将预测向量转换为标签，predictions是one-hot编码，需要argmax获取最大概率对应的类别
    predicted_classes = np.argmax(prediction, axis=1)
    print('Predictions:', prediction, predicted_classes)
    accuracy = np.sum(predicted_classes == labels) / \
        len(nd_array_images)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return predicted_classes


def draw_image_prediction(nd_array_images, predictions, num=10):
    print(nd_array_images.shape, predictions.shape)
    # 随机选择指定数量的图片及其预测结果
    random_indices = np.random.choice(
        range(len(nd_array_images)), size=num, replace=False)
    sample_images = nd_array_images[random_indices]
    sample_predictions = predictions[random_indices]

    # 创建一个figure对象用于多子图展示
    __fig, axs = plt.subplots(nrows=num, ncols=1,
                              figsize=(10, num * 2))

    for i, (image, pred) in enumerate(zip(sample_images, sample_predictions)):
        ax = axs[i]  # 获取当前的Axes对象
        if image.shape[-1] == 1:
            ax.imshow((image * 255).astype(np.uint8), cmap='gray')
        else:
            ax.imshow(image.astype(np.uint8))

        ax.set_title(f"Image {i + 1}: Predicted Class - {pred}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
