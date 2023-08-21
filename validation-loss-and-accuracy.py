def plot_training_history(history, start_epoch):
    # Get training and validation loss and accuracy histories
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']

    # Get the total number of epochs training was run for
    total_epochs = len(training_accuracy) + start_epoch

    # Create a list of epoch numbers
    epoch_nums = range(start_epoch + 1, total_epochs + 1)

    # Find the epoch at which validation loss was minimum and validation accuracy was maximum
    min_val_loss_epoch = np.argmin(validation_loss) + start_epoch + 1
    max_val_accuracy_epoch = np.argmax(validation_accuracy) + start_epoch + 1

    # Create a plot for loss
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_nums, training_loss, 'r', label='Training Loss')
    plt.plot(epoch_nums, validation_loss, 'g', label='Validation Loss')
    plt.scatter(min_val_loss_epoch, validation_loss[min_val_loss_epoch - start_epoch - 1], s=150, c='blue', label=f'Lowest Validation Loss at Epoch {min_val_loss_epoch}')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Create a plot for accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epoch_nums, training_accuracy, 'r', label='Training Accuracy')
    plt.plot(epoch_nums, validation_accuracy, 'g', label='Validation Accuracy')
    plt.scatter(max_val_accuracy_epoch, validation_accuracy[max_val_accuracy_epoch - start_epoch - 1], s=150, c='blue', label=f'Highest Validation Accuracy at Epoch {max_val_accuracy_epoch}')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

# Plot the training history
plot_training_history(history, start_epoch=0)
