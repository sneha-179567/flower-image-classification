import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the saved model
model = tf.keras.models.load_model("flower_model.h5")

# Flower class names
classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load the test image
img = image.load_img("test.jpg", target_size=(180, 180))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

predicted_class = classes[np.argmax(prediction)]

print("Predicted Flower:", predicted_class)