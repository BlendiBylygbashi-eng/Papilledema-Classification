import cv2
import numpy as np

def isolate_optic_disk(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to isolate bright areas
    _, thresholded = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY)

    # Perform morphological operations to remove small noise
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations = 2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours based on area
    min_area = 500  # This value will depend on the size of the optic disc in your images
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Create an empty mask to draw the contours
    mask = np.zeros_like(gray)

    # Draw the contours on the mask
    cv2.drawContours(mask, large_contours, -1, (255), thickness=cv2.FILLED)

    # Bitwise-and with the original image
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return result
import matplotlib.pyplot as plt

def display_images(original, result):
    # Display the original image and the result side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Result after Optic Disk Isolation')
    plt.show()

# Specify the path to your image
image_path = "/content/drive/MyDrive/Test_Images_New/Normal/im0235.ppm"

# Read the image using OpenCV
original_image = cv2.imread(image_path)

# Apply the function to isolate the optic disc
isolated_image = isolate_optic_disk(original_image)

# Display the original image and the result
display_images(original_image, isolated_image)
