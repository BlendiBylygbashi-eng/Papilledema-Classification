# Start training the model
history = model.fit(
    # Training data generator
    x=train_gen,  

    # Number of epochs to train
    epochs=epochs, 

    # Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch)
    verbose=1, 

    # List of callbacks to apply during training
    callbacks=callbacks, 

    # Validation data generator
    validation_data=valid_gen, 

    # Number of steps (batches of samples) to yield from validation_data generator before stopping at the end of every epoch. 
    # It will default to len(validation_data) if not specified.
    validation_steps=None,  

    # Whether to shuffle the order of the batches at the beginning of each epoch.
    shuffle=False,  

    # Epoch at which to start training (useful for resuming a previous training run)
    initial_epoch=0
)
