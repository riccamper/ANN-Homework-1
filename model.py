import os
import tensorflow as tf
from keras import backend as K


class model:

    # F1 Score
    def f1(y_true, y_pred):
        # Recall
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        # Precision
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    # Init
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(
            path, 'SubmissionModel'), custom_objects={'f1': self.f1})

    # Predict
    def predict(self, X):
        # Insert your preprocessing here
        X_test = X/255.
        out = self.model.predict(X_test)
        out = tf.argmax(out, axis=-1)
        return out
