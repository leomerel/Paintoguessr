# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np

import datetime

import preprocess
import plot


class Cnn:
    def __init__(self):
        self.class_names, self.images, self.labels = preprocess.createSets()
        self.train_images, self.test_images, self.train_labels, self.test_labels = preprocess.splitData(self.images, self.labels, test_size=0.3, shuffle=True)
        print(len(self.class_names), self.class_names)

        self.train_images = preprocess.formatImages(self.train_images)
        self.test_images = preprocess.formatImages(self.test_images)

    def create_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='sigmoid'),
            tf.keras.layers.Dense(64, activation='sigmoid'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.class_names))
        ])

        self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        self.model.fit(self.train_images, self.train_labels, epochs=10)

        return self.model

    def save_model(self, model):
        model_path = "../models/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model.save(model_path)

    def load_model(self, name = "2021-11-08_18-46-00(16-10000)"):
        self.model = tf.keras.models.load_model("../models/" + name)
        return self.model

    def test_model(self, model):
        test_loss, test_acc = model.evaluate(self.test_images, self.test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)

    def get_predictions(self):
        probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        predictions = probability_model.predict(self.test_images)
        print(predictions[0])
        print(np.argmax(predictions[0]))
        print(self.test_labels[0])

        return predictions

    def get_prediction(self, image):
        probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])

        prediction = probability_model(image)[0]

        dict = {}
        for i in range(len(self.class_names)):
            dict[self.class_names[i]] = round(prediction[i].numpy() * 100, 2)

        dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1])}

        print(dict)
        print("best: ", self.class_names[np.argmax(prediction[0])])

        return dict


if __name__ == '__main__':
    cnn = Cnn()
    model = cnn.create_model()
    cnn.save_model(model)
    cnn.test_model(model)
    predictions = cnn.get_predictions()
    plot.plot_test_model(predictions, cnn.test_labels, cnn.test_images, cnn.class_names)