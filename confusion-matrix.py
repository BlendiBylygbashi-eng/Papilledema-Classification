def evaluate_model(test_data_generator):
    # Get true labels and number of classes from test data generator
    true_labels = np.array(test_data_generator.labels)
    class_names = list(test_data_generator.class_indices.keys())
    num_classes = len(class_names)

    # Make predictions with the model
    predictions = model.predict(test_data_generator, verbose=1)
    predicted_indices = np.argmax(predictions, axis=-1)

    # Count errors and compute accuracy
    num_errors = np.sum(predicted_indices != true_labels)
    num_tests = len(predictions)
    accuracy = (1 - num_errors / num_tests) * 100

    print(f'There were {num_errors} errors in {num_tests} tests for an accuracy of {accuracy:.2f}%')

    if num_classes <= 30:
        # Display confusion matrix
        confusion_mtx = confusion_matrix(true_labels, predicted_indices)
        plt.figure(figsize=(12, 8))
        sns.heatmap(confusion_mtx, annot=True, fmt='g', cmap='Blues', cbar=False)
        plt.xticks(ticks=np.arange(num_classes)+0.5, labels=class_names, rotation=90)
        plt.yticks(ticks=np.arange(num_classes)+0.5, labels=class_names, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    # Display classification report
    class_report = classification_report(true_labels, predicted_indices, target_names=class_names, digits=4)
    print("Classification Report:\n----------------------\n", class_report)

    return num_errors, num_tests

# Evaluate the model
num_errors, num_tests = evaluate_model(test_gen)
