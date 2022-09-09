from tensorflow.keras.layers import Dense, Flatten,Conv2D , Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.layers import MaxPooling2D
import tensorflow as tf
import matplotlib.pyplot as plt

train_dir = 'Data/train'
test_dir = 'Data/test'
train_datagen = ImageDataGenerator(
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(
                                  rescale = 1./255)
# ------------------------------
training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 )
test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            )
val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        subset='validation')
# -------------------------------
print('building model')
model=tf.keras.Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=128,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(1024,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(512,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(256,activation="relu"))
model.add(Dense(3,activation="softmax"))
model.summary()
METRICS = [
    'accuracy','categorical_crossentropy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

model.compile(
    loss='categorical_crossentropy',
  optimizer='adam',
  metrics=METRICS
)
class_weight = {
    0:100.,
    1:75.,
    2:30.
}
print('training model')

r = model.fit(training_set,
  validation_data = val_generator,
  epochs=25,
  steps_per_epoch=32,
  class_weight=class_weight)
# ---------------------------

model.evaluate(test_set)
