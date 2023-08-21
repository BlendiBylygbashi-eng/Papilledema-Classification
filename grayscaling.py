def grayscale_conversion(image):
    # Ensure the image data type is uint8 for accurate conversion
    image = image.astype(np.uint8)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Stack the grayscale image 3 times to create a 3-channel image
    stacked_gray = np.stack((gray,)*3, axis=-1)
    
    # Clip the pixel values to the valid range [0, 255]
    stacked_gray = np.clip(stacked_gray, 0, 255)

    # Return the image after converting back to uint8
    return stacked_gray.astype(np.uint8)
