def trim(df, max_samples, min_samples, column):
    # Copy the dataframe to avoid inplace modifications
    df = df.copy()
    
    # Initialize an empty dataframe with the same columns as the input dataframe
    trimmed_df = pd.DataFrame(columns=df.columns)
    
    # Group the dataframe by the specified column
    groups = df.groupby(column)

    # Iterate over each unique value in the specified column
    for label in df[column].unique(): 
        group = groups.get_group(label)
        count = len(group)    

        # If count exceeds max_samples, randomly sample max_samples from the group
        if count > max_samples:
            sampled_group = group.sample(n=max_samples, random_state=123)
            trimmed_df = pd.concat([trimmed_df, sampled_group], axis=0)
        else:
            # If count is greater than or equal to min_samples, keep the entire group
            if count >= min_samples:
                trimmed_df = pd.concat([trimmed_df, group], axis=0)

    print(f'After trimming, the maximum samples in any class is now {max_samples} and the minimum samples in any class is {min_samples}')

    return trimmed_df

# Set the maximum and minimum samples per class
max_samples = 236
min_samples = 226
column = 'labels'

# Trim the training dataframe to the specified number of samples per class
train_df = trim(train_df, max_samples, min_samples, column)
