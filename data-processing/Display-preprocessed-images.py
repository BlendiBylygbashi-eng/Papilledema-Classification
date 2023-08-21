import matplotlib.pyplot as plt
import numpy as np


def show_image_samples(gen):
    """
    Displays a batch of image samples and their labels.

    Parameters:
        gen (Generator): An ImageDataGenerator object.
    """
    # Get class labels
    t_dict = gen.class_indices
    classes = list(t_dict.keys())

    # Get a batch of images and labels
    images, labels = next(gen)

    # Define the size of the plot
    plt.figure(figsize=(20, 20))

    # Determine the number of images to display (max 25)
    length = len(labels)
    num_images = min(length, 25)  # Show a maximum of 25 images

    # Plot each image with its class label
    for i in range(num_images):  
        plt.subplot(5, 5, i + 1)  # Create a 5x5 grid of subplots

        # Normalize image pixel values to [0, 1] for correct display
        image = images[i] / 255

        # Display the image
        plt.imshow(image)

        # Get the class label for the image
        index = np.argmax(labels[i])
        class_name = classes[index]

        # Display the class label
        plt.title(class_name, color='blue', fontsize=14)

        # Hide axes for clarity
        plt.axis('off')

    # Display the plot with image samples
    plt.show()

# Show image samples from the training generator
show_image_samples(train_gen)
