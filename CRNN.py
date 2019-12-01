from keras import layers, Sequential, Model, datasets
from keras.layers import Input
import keras
import tensorflow as tf
import numpy as np
x_train = np.load(r'data/train/total_image_137.npy')
y_train = np.load(r'data/train/total_label_137.npy').reshape(-1,1)
x_test = np.load(r'data/test/total_image_15.npy')
y_test = np.load(r'data/test/total_label_15.npy').reshape(-1,1)
model1 = keras.models.load_model(r'model/total_model_1000.model')
x_train = model1.predict(x_train)
x_test = model1.predict(x_test)
mean = 0
var = []
for i in range(6):
    callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs/{}/'.format(i))
    ]

    model = Sequential([
        keras.layers.Conv2D(128, 3, padding='same',input_shape=(64, 64, 24)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(256, 3, padding='same'),
        keras.layers.MaxPooling2D(),
        keras.layers.Permute((3,1,2)),
        keras.layers.TimeDistributed(keras.layers.Flatten()),
        keras.layers.GRU(24),
        keras.layers.Dense(3, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=10, epochs=20, validation_split=0.1, callbacks=callbacks)
    score = model.evaluate(x_test, y_test)
    mean = mean + score[1]
    var.append(score[1])
print(mean/6)
print(np.var(var))


