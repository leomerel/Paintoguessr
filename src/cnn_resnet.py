# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np

import datetime

import preprocess
import plot

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


class Cnn:
    def __init__(self):
        self.class_names, self.images, self.labels = preprocess.createSets()
        self.train_images, self.test_images, self.train_labels, self.test_labels = preprocess.splitData(self.images,
                                                                                                        self.labels,
                                                                                                        test_size=0.3,
                                                                                                        shuffle=True)
        print(len(self.class_names), self.class_names)

        self.train_images = preprocess.formatImages(self.train_images)
        self.test_images = preprocess.formatImages(self.test_images)

    def create_model(self):
        self.model = tf.keras.applications.resnet50()

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        self.model.fit(self.train_imaes, self.train_labels, epochs=10)

        return self.model

    def save_model(self, model):
        model_path = "../models/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model.save(model_path)

    def load_model(self, name="2021-11-08_18-46-00(16-10000)"):
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

        prediction = probability_model(image)
        print(prediction)
        print(self.class_names[np.argmax(prediction[0])])

        return prediction[0]

# if __name__ == '__main__':
#     cnn = Cnn()
#     model = cnn.load_model()
#     cnn.test_model(model)
#     predictions = cnn.get_predictions()
#     plot.plot_test_model(predictions, cnn.test_labels, cnn.test_images, cnn.class_names)
