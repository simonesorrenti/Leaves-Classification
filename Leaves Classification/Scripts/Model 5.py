import os
from keras import backend as K
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from matplotlib import pyplot as plt

tfk = tf.keras
tfkl = tf.keras.layers

# Random seed for reproducibility
seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# Labels
labels = ['Apple',  # 0
          'Blueberry',  # 1
          'Cherry',  # 2
          'Corn',  # 3
          'Grape',  # 4
          'Orange',  # 5
          'Peach',  # 6
          'Pepper',  # 7
          'Potato',  # 8
          'Raspberry',  # 9
          'Soybean',  # 10
          'Squash',  # 11
          'Strawberry',  # 12
          'Tomato']  # 13

# Datagenerator with data augmentation, validation split and preprocessing of Inception V3
dataset_gen = ImageDataGenerator(rotation_range=170,
                                 width_shift_range=0.3,
                                 height_shift_range=0.3,
                                 brightness_range=[1, 5],
                                 zoom_range=[0.5, 1.5],
                                 fill_mode="constant",
                                 cval=0.,
                                 validation_split=0.2,
                                 preprocessing_function=preprocess_input)

train_gen = dataset_gen.flow_from_directory(directory='../input/dataset/training',
                                            target_size=(256, 256),
                                            color_mode='rgb',
                                            classes=None,  # can be set to labels
                                            class_mode='categorical',
                                            batch_size=32,
                                            shuffle=True,
                                            seed=seed,
                                            subset='training')

valid_gen = dataset_gen.flow_from_directory(directory='../input/dataset/training',
                                            target_size=(256, 256),
                                            color_mode='rgb',
                                            classes=None,  # can be set to labels
                                            class_mode='categorical',
                                            batch_size=32,
                                            shuffle=False,
                                            seed=seed,
                                            subset='validation')

# Download the InceptionV3 model
inception = InceptionV3(
    include_top=False,
    weights="imagenet"
 )
inception.trainable = True
inception.summary()

# Fine Tuning
x = inception.output
x = tfkl.GlobalAveragePooling2D()(x)
x = tfkl.Dropout(0.2, seed=seed)(x)
x = tfkl.Dense(1024, activation='relu', kernel_initializer=tfk.initializers.GlorotUniform(seed))(x)
x = tfkl.Dropout(0.2, seed=seed)(x)
predictions = tfkl.Dense(14, activation='softmax', kernel_initializer=tfk.initializers.GlorotUniform(seed))(x)

# Connect input and output through the Model class
model = tfk.Model(inputs=inception.inputs, outputs=predictions)

# Compile the model
model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(learning_rate=1e-3),
              metrics=['accuracy', f1_m])
model.summary()

# Compute the class weights
from collections import Counter
counter = Counter(train_gen.classes)
max_val = float(max(counter.values()))
class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}

# Train the model
model_history = model.fit(
    x=train_gen,
    epochs=50,
    class_weight=class_weights,
    validation_data=valid_gen,
    callbacks=[tfk.callbacks.EarlyStopping(monitor='val_f1_m', mode='max', patience=10, restore_best_weights=True)]
).history

model.save('InceptionV3_Fine_Tuning_F1_classweight')

# Plot the training
plt.figure(figsize=(15, 5))
plt.plot(model_history['loss'], alpha=.3, color='#4D61E2', linestyle='--')
plt.plot(model_history['val_loss'], label='Transfer Learning', alpha=.8, color='#4D61E2')
plt.legend(loc='upper left')
plt.title('Categorical Crossentropy')
plt.grid(alpha=.3)
plt.figure(figsize=(15, 5))
plt.plot(model_history['accuracy'], alpha=.3, color='#ff7f0e', linestyle='--')
plt.plot(model_history['val_accuracy'], label='Standard', alpha=.8, color='#ff7f0e')
plt.legend(loc='upper left')
plt.title('Accuracy')
plt.grid(alpha=.3)
plt.show()
plt.plot(model_history['f1_m'], alpha=.3, color='#ff7f0e', linestyle='--')
plt.plot(model_history['val_f1_m'], label='Standard', alpha=.8, color='#ff7f0e')
plt.legend(loc='upper left')
plt.title('F1 measure')
plt.grid(alpha=.3)
plt.show()
