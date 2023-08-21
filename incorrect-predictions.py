import matplotlib.pyplot as plt
import os

# Get the predictions for the test set
predictions = model.predict(test_gen)

# Get the class with the highest probability for each sample
pred_labels = np.argmax(predictions, axis=1)

# Get the true labels
true_labels = test_gen.labels

# Get the indices of the wrongly classified images
wrong_indices = np.nonzero(pred_labels != true_labels)[0]

# Get the filepaths of the wrongly classified images
wrong_images = [test_gen.filepaths[i] for i in wrong_indices]

# List to hold filenames of wrongly classified images
wrong_filenames = []

# For each wrongly classified image
for i, image_path in enumerate(wrong_images):
    # Load the image
    img = plt.imread(image_path)
    
    # Get the filename and the parent folder name
    folder_name = os.path.basename(os.path.dirname(image_path))
    filename = os.path.basename(image_path)

    # Combine the folder name and filename
    full_name = os.path.join(folder_name, filename)

    # Append full name to the list
    wrong_filenames.append(full_name)
    
    # Show the image
    plt.figure()
    plt.imshow(img)
    plt.title(f"True: {classes[true_labels[wrong_indices[i]]]}, Pred: {classes[pred_labels[wrong_indices[i]]]}")
    plt.show()

    # Optionally, you can stop after a certain number of images
    if i > 10:
        break

# Print out the full names of the wrongly classified images
print("Files of wrongly classified images:")
for full_name in wrong_filenames:
    print(full_name)
