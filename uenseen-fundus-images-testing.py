from sklearn.metrics import accuracy_score
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

def predict_class(model, image_path):
    # Load the image
    img = load_img(image_path, target_size=(240, 240))  # adjust target size to match model's expected input size
    
    # Convert the image to numpy array and apply Gaussian color subtraction
    img_array = img_to_array(img)
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

def evaluate_model(model, test_folder, classes):
    true_labels = []
    predicted_labels = []

    # Iterate over each subfolder corresponding to a class in the test folder
    for class_folder_name in os.listdir(test_folder):
        class_folder_path = os.path.join(test_folder, class_folder_name)

        # Check if it's a folder
        if os.path.isdir(class_folder_path):

            # Iterate over each image in the subfolder
            for image_name in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_name)

                # Predict the class of the image
                predicted_class = predict_class(model, image_path)

                # Add the true class and predicted class to their respective lists
                true_labels.append(class_folder_name)
                predicted_labels.append(predicted_class)

                # Print out the actual and predicted classes
                print(f"Image: {image_name}")
                print(f"Actual class: {class_folder_name}")
                print(f"Predicted class: {predicted_class}")
                print("-------------------------------")

    # Compute the test accuracy
    test_accuracy = accuracy_score(true_labels, predicted_labels)

    return test_accuracy


# Define the test folder and the class list
test_folder = "/content/drive/MyDrive/Test_Images_New"
classes = ['Normal', 'Papilledema', 'Pseudopapilledema']

# Call the function and print the accuracy
print(evaluate_model(model, test_folder, classes))
