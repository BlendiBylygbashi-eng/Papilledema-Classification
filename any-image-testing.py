from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input


def predict_class(model, image_path):
    # Load the image
    img = load_img(image_path, target_size=(224, 224))  # adjust target size to match your model input
    
    # Convert the image to numpy array
    img_array = img_to_array(img)
    
    # Apply Gaussian color subtraction
    img_array = gaussian_color_subtraction(img_array)
    
    # Expand the dimensions of the image as we are predicting on a single instance
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image
    img_preprocessed = preprocess_input(img_batch)
    
    # Use the model to predict the class
    prediction = model.predict(img_preprocessed)
    
    # Get the class with highest probability
    predicted_class = np.argmax(prediction)
    
    # Convert class indices back to class labels and return
    return classes[predicted_class]

# Testing
image_path = "/content/drive/MyDrive/Test_Images_New/Papilledema/original.jpg"
print(predict_class(model, image_path))
