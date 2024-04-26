# Dataset Split 
import pandas as pd
import numpy as np

# Load the file into a pandas DataFrame
df = pd.read_excel(r'C:/Users/jaco/OneDrive/Desktop/PHD/Python/Data/MULTIWOZ Data.xlsx')

# Get the unique conversation IDs
conversation_ids = df['Conversation'].unique()

# Shuffle the conversation IDs
np.random.shuffle(conversation_ids)

# Split the conversation IDs into three groups
group1_ids = conversation_ids[:3479]
group2_ids = conversation_ids[3479:3479*2]
group3_ids = conversation_ids[3479*2:]

# Function to split a group of IDs into train, validation, and test sets and save them
def split_and_save(ids, group_name):
    train_ids = ids[:int(len(ids) * 0.7)]
    val_ids = ids[int(len(ids) * 0.7):int(len(ids) * 0.9)]
    test_ids = ids[int(len(ids) * 0.9):]

    train_df = df[df['Conversation'].isin(train_ids)]
    val_df = df[df['Conversation'].isin(val_ids)]
    test_df = df[df['Conversation'].isin(test_ids)]

    base_path = 'C:/Users/jaco/OneDrive/Desktop/PHD/Python/Train_Validate_Test_Files//'
    
    train_df.to_excel(base_path + group_name + '_Train.xlsx', index=False)
    val_df.to_excel(base_path + group_name + '_Validate.xlsx', index=False)
    test_df.to_excel(base_path + group_name + '_Test.xlsx', index=False)

# Split and save each group
split_and_save(group1_ids, 'Group1')
split_and_save(group2_ids, 'Group2')
split_and_save(group3_ids, 'Group3')
