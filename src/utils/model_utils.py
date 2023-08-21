
import tensorflow as tf


def create_model():
    res = tf.keras.applications.ResNet50(include_top=False, weights='imagenet',
                                         input_shape=(224, 224, 3))

    for layer in res.layers:
        layer.trainable = False

    flat1 = tf.keras.layers.Flatten()(res.layers[-1].output)
    class1 = tf.keras.layers.Dense(256, activation='relu')(flat1)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(class1)
    model = tf.keras.models.Model(inputs=res.inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy', tf.keras.metrics.AUC()])
    return model
