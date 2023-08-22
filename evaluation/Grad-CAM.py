import cv2
import tensorflow as tf 
from tensorflow import keras
import numpy as np 

# Use the pre-trained model, called 'model'

# Build Grad-CAM Model
grad_cam_model = keras.models.Model([model.inputs], [model.get_layer('block6e_project_conv').output, model.output]) # Replace with a convolution layer of your choosing

# Fetch a batch of images and labels from train_gen
images, labels = next(train_gen)

# Use the first image from the batch for Grad-CAM
cam_image = images[0]

# Resize the image to the expected input shape (224, 224)
cam_image = cv2.resize(cam_image, (224, 224))

# Check if the image is already normalized to [0,1], if not normalize it
if np.max(cam_image) > 1:
    cam_image = cam_image / 255.0

# Add batch dimension
cam_image = np.expand_dims(cam_image, axis=0) 

# Record operations for automatic differentiation
with tf.GradientTape() as cam_tape:
    cam_convOutputs, cam_predictions = grad_cam_model(tf.cast(cam_image, tf.float32))
    cam_loss = cam_predictions[:, tf.argmax(cam_predictions[0])]

# Derive the gradients
cam_grads = cam_tape.gradient(cam_loss, cam_convOutputs)

# Compute guided gradients
cam_castConvOutputs = tf.cast(cam_convOutputs > 0, "float32")
cam_castGrads = tf.cast(cam_grads > 0, "float32")
cam_guidedGrads = cam_castConvOutputs * cam_castGrads * cam_grads

# Remove the batch dimension
cam_convOutputs = cam_convOutputs[0]
cam_guidedGrads = cam_guidedGrads[0]

# Compute the average of the gradients spatially
cam_weights = tf.reduce_mean(cam_guidedGrads, axis=(0, 1))

# Create a new tensor with the same shape as the gradients
cam_cam = tf.reduce_sum(tf.multiply(cam_weights, cam_convOutputs), axis=-1)

# Resize heatmap to match the input image dimensions
(w, h) = (cam_image.shape[2], cam_image.shape[1])
cam_heatmap = cv2.resize(cam_cam.numpy(), (w, h))

# Normalize the heatmap such that all values lie in the range
cam_heatmap = np.maximum(cam_heatmap, 0)
cam_heatmap /= np.max(cam_heatmap)
cam_heatmap = cv2.applyColorMap(np.uint8(255*cam_heatmap), cv2.COLORMAP_JET)

# Get the original image from the batch dimension
cam_original_image = cam_image[0]

# Ensure the image is in the range [0, 255] for visualization
cam_original_image = np.uint8(255 * cam_original_image)

# Convert the original image color format from RGB to BGR 
cam_original_image = cv2.cvtColor(cam_original_image, cv2.COLOR_RGB2BGR)

# Superimpose the heatmap on original image
cam_output_image = cv2.addWeighted(cam_original_image, 0.5, cam_heatmap, 0.5, 0)

import matplotlib.pyplot as plt

# Convert BGR image to RGB
cam_output_image_rgb = cv2.cvtColor(cam_output_image, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(cam_output_image_rgb)
plt.axis('off')  # remove axes for visual appeal
plt.show()

plt.imshow(cam_original_image)
plt.axis('off')  # remove axes for visual appeal
plt.show()
