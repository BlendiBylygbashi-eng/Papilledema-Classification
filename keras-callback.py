# This is a custom Keras callback that allows the user to halt training or continue for additional epochs.
class LrAsk(keras.callbacks.Callback):
    # Initializes the callback. The model, total epochs, and the epoch to ask are passed as arguments.
    def __init__(self, model, epochs, ask_epoch): 
        super().__init__()
        self.model = model
        self.epochs = epochs
        self.ask_epoch = ask_epoch
        self.ask = True 
        self.lowest_vloss = np.inf
        self.best_weights = self.model.get_weights() 
        self.best_epoch = 1

    # This function is called at the beginning of training.
    def on_train_begin(self, logs=None):
        # If the ask epoch is not within the total epochs, disable asking.
        if self.ask_epoch == 0 or self.ask_epoch >= self.epochs or self.epochs == 1:
            self.ask = False 
            print('Adjusted ask_epoch. Training will run for', self.epochs, 'epochs')
        else:
            # Otherwise, print the ask epoch.
            print('Training will proceed until epoch', self.ask_epoch) 
            print('Enter H to halt training or an integer for additional epochs at ask_epoch')
        # Record the start time of training.
        self.start_time = time.time() 

    # This function is called at the end of training.
    def on_train_end(self, logs=None):   
        # Set the model weights to the best weights found during training.
        self.model.set_weights(self.best_weights) 
        # Print the total time taken for training.
        elapsed_time = time.time() - self.start_time           
        print(f'Training completed in: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
        
    # This function is called at the end of each epoch.
    def on_epoch_end(self, epoch, logs=None):  
        v_loss = logs.get('val_loss') 
        # If the validation loss is lower than the lowest recorded validation loss,
        # update the lowest validation loss and save the current weights as the best weights.
        if v_loss < self.lowest_vloss:
            self.lowest_vloss = v_loss
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch + 1
            print (f'Saving weights from epoch {self.best_epoch} as best weights')

        # If it is the ask epoch, ask the user whether to halt training or continue for additional epochs.
        if self.ask and epoch + 1 == self.ask_epoch:
            ans = input('\nEnter H to end training or an integer for additional epochs: ')
            
            # If the user inputs 'H', halt training.
            if ans in ['H', 'h', '0']: 
                self.model.stop_training = True 
                print ('Training halted at epoch', epoch+1)
            else: 
                # Otherwise, add the input number to the ask epoch.
                self.ask_epoch += int(ans)
                if self.ask_epoch > self.epochs:
                    print('Cannot train for', self.ask_epoch, 'epochs. Will train for', self.epochs)
                else:
                    # Ask the user whether to change the learning rate.
                    lr = float(tf.keras.backend.get_value(self.model.optimizer.lr)) 
                    ans = input(f'Current LR is {lr}. Press Enter to keep or input a new LR: ')
                    
                    # If the user presses Enter, keep the current learning rate.
                    if ans =='':
                        print ('Keeping current LR', lr)
                    else:
                        # Otherwise, change the learning rate to the input number.
                        new_lr = float(ans)
                        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                        print('Changing LR to', new_lr)

# Define the number of epochs
epochs = 40

# Initialize the ReduceLROnPlateau callback if you want to use it
# This callback reduces the learning rate when the validation loss stops improving
rlronp = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)

# List of callbacks
callbacks = [rlronp]
