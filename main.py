from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model



train_dir =r"C:\Users\krish\ml\data\data\train"
test_dir = r"C:\Users\krish\ml\data\data\test"
val_dir = r"C:\Users\krish\ml\data\data\val"

train_datagen = ImageDataGenerator( zoom_range=0.2,
                            width_shift_range=0.2,height_shift_range=0.2)


train_dg = train_datagen.flow_from_directory(train_dir,
                                    class_mode = "categorical",
                                    target_size = (299, 299),
                                    batch_size = 128,
                                    shuffle = True,
                                    seed = 42)

val_datagen = ImageDataGenerator()
validation_dg = val_datagen.flow_from_directory(val_dir,
                                      class_mode = "categorical",
                                      target_size = (299, 299),
                                      batch_size = 128,
                                      shuffle = False,
                                      seed = 42)

testing_dg = val_datagen.flow_from_directory(test_dir,
                                      class_mode = "categorical",
                                      target_size = (299, 299),
                                      batch_size = 128,
                                      shuffle = False,
                                      seed = 42)


from tensorflow.keras.applications import EfficientNetB0

base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(299, 299, 3))

# Freeze the base model layers (optional, can be fine-tuned later)
for layer in base_model.layers:
  layer.trainable = False

# Add custom layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.25)(x)
predictions = Dense(10, activation="softmax")(x)  # Adjust output size for 500 species

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

opt = Adam(learning_rate=0.005)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

history = model.fit(
      train_dg,
      epochs=10,
      validation_data = validation_dg,
    callbacks=[
        EarlyStopping(monitor = "val_loss", # watch the val loss metric
                               patience = 3,
                               restore_best_weights = True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, mode='min')
    ]
)
model.save('cnn_model.h5')
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

# class_labels = ['air hockey', 'ampute football', 'archery', 'arm wrestling', 'axe throwing', 'balance beam', 'barell racing', 'baseball', 'basketball', 'baton twirling']
#
# # Load and preprocess a random image
#  # Replace with your image path
# def predict(img_path):
#   model1 = load_model('cnn_model.h5')
#   img = image.load_img(img_path, target_size=(299, 299))
#   img_array = image.img_to_array(img)
#   img_array = np.expand_dims(img_array, axis=0)
#   img_array = preprocess_input(img_array)
#
# # Make predictions
#   predictions = model1.predict(img_array)
#
#   predicted_class_index = np.argmax(predictions, axis=1)[0]
#   predicted_class_label = class_labels[predicted_class_index]
#   predicted_class_probability = predictions[0][predicted_class_index]
#
#   return (f"{predicted_class_label.capitalize()} and its probability is {predicted_class_probability*100:.2f}%")


