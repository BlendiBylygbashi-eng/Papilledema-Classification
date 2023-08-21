import cv2
import numpy as np

def gaussian_color_subtraction(image):
    # Ensure the image data type is float for accurate subtraction and division operations
    image = image.astype(float)

    # Compute the Gaussian blur of the image
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=30)
    
    # Subtract the blurred image from the original image
    subtracted = cv2.subtract(image, blurred)

    # Clip the pixel values to the valid range [0, 255]
    subtracted = np.clip(subtracted, 0, 255)

    # Return the image after converting back to uint8
    return subtracted.astype(np.uint8)

