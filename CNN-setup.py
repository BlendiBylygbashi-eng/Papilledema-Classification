import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model


    # EfficientNetB3


# Define the size of your input images
img_size = (224, 224)
img_shape = (img_size[0], img_size[1], 3)

# Create the base pre-trained model using EfficientNetB3
base_model = tf.keras.applications.EfficientNetB3(
    include_top=False,  # Do not include the top (last) fully connected layer
    weights="imagenet",  # Use pre-trained weights from ImageNet
    input_shape=img_shape,  # Define the input shape
    pooling='max'  # Use max pooling for the output of the last convolutional layer
)

# Make all layers in the base model trainable
base_model.trainable = True

# Add custom layers on top of the base model
x = base_model.output
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)  # Add a batch normalization layer
x = Dense(256, 
          kernel_regularizer=regularizers.l2(l=0.016),  # Apply L2 regularization to the weights
          activity_regularizer=regularizers.l1(0.006),  # Apply L1 regularization to the activation
          bias_regularizer=regularizers.l1(0.006),  # Apply L1 regularization to the biases
          activation='relu')(x)  # Add a dense layer with ReLU activation function
x = Dropout(rate=.4, seed=123)(x)  # Add a dropout layer for regularization
output = Dense(class_count, activation='softmax')(x)  # Add the output layer with softmax activation function

# Combine the base model with the custom layers
model = Model(inputs=base_model.input, outputs=output)

# Compile the model with an optimizer, loss function, and performance metric
model.compile(
    optimizer=Adamax(learning_rate=0.001),  # Use the Adamax optimizer
    loss='categorical_crossentropy',  # Use categorical cross entropy as the loss function
    metrics=['accuracy']  # Track accuracy during the training
)
