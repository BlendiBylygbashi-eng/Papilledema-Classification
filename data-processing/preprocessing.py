from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the directory where the script is running
working_dir = r'./' 

# Define the desired size of the images for model input
img_size = (240, 240)

# Define the batch size for training, suitable for EfficientetB3 model
batch_size = 30

# Initialize an ImageDataGenerator for augmenting training images
trgen = ImageDataGenerator(horizontal_flip=True, rotation_range=20, width_shift_range=.2, height_shift_range=.2, zoom_range=.2)

# Initialize another ImageDataGenerator for validation and testing (without augmentation)
t_and_v_gen = ImageDataGenerator()

# Print a placeholder message for train generator loading
print('{0:70s} for train generator'.format(' '), '\r', end='')

# Generate augmented images for training from the train_df DataFrame
train_gen = trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)

# Print a placeholder message for validation generator loading
print('{0:70s} for valid generator'.format(' '), '\r', end='')

# Generate images for validation from the valid_df DataFrame
valid_gen = t_and_v_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)

# Calculate batch size for test set such that each sample in the test set is used exactly once
length = len(test_df)
test_batch_size = sorted([int(length/n) for n in range(1, length+1) if length % n == 0 and length/n <= 80], reverse=True)[0]  
test_steps = int(length/test_batch_size)

# Print a placeholder message for test generator loading
print('{0:70s} for test generator'.format(' '), '\r', end='')

# Generate images for testing from the test_df DataFrame
test_gen = t_and_v_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size)

# Extract some useful information from the train generator
classes = list(train_gen.class_indices.keys())
class_indices = list(train_gen.class_indices.values())
class_count = len(classes)
labels = test_gen.labels

print(f'Test batch size: {test_batch_size}, Test steps: {test_steps}, Number of classes: {class_count}')
