# define the directory where the data is stored
sdir = r'/content/drive/MyDrive/Papilledema_Data_2'

# Initialize lists for storing filepaths and labels
filepaths = []
labels = [] 

# Get a sorted list of all subdirectories in the main directory
classlist = sorted(os.listdir(sdir))  

# Iterate over each subdirectory and append its filepaths and labels to respective lists
for klass in classlist:
    classpath = os.path.join(sdir, klass)
    for f in os.listdir(classpath):        
        filepaths.append(os.path.join(classpath, f))
        labels.append(klass)

# Convert the lists into pandas Series
Fseries = pd.Series(filepaths, name='filepaths')
Lseries = pd.Series(labels, name='labels')

# Concatenate the series into a dataframe
df = pd.concat([Fseries, Lseries], axis=1)

# Split the dataframe into training, validation, and testing sets
train_df, dummy_df = train_test_split(df, train_size=.8, shuffle=True, random_state=123, stratify=df['labels'])
valid_df, test_df = train_test_split(dummy_df, train_size=.5, shuffle=True, random_state=123, stratify=dummy_df['labels'])

print(f'Train set length: {len(train_df)}, Test set length: {len(test_df)}, Validation set length: {len(valid_df)}')

# Get the number of unique classes and their counts
class_count = train_df['labels'].nunique()
print(f'The number of classes in the dataset is: {class_count}')

# Group the dataframe by labels and get their counts
class_counts = train_df['labels'].value_counts()

# Display the class-wise image counts
print('CLASS:\t\tIMAGE COUNT:')
for idx, val in class_counts.iteritems():
    print(f'{idx}:\t{val}')

# Get the classes with minimum and maximum number of images
min_class = class_counts.idxmin()
max_class = class_counts.idxmax()
print(f'{max_class} has the most images={class_counts[max_class]}, {min_class} has the least images={class_counts[min_class]}')

# Compute the average height and width of a random sample of training images
heights, widths = [], []
train_df_sample = train_df.sample(n=100, random_state=123)

for filepath in train_df_sample['filepaths']:
    img = plt.imread(filepath)
    heights.append(img.shape[0])
    widths.append(img.shape[1])

avg_height = sum(heights) / len(heights)
avg_width = sum(widths) / len(widths)

print(f'Average height= {avg_height}, Average width= {avg_width}, Aspect ratio= {avg_height / avg_width}')
