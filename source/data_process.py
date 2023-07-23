# import libraries
import os
import pandas as pd
import yaml
import sys
from sklearn.model_selection import train_test_split


# parameters

split_ratio = yaml.safe_load(open('params.yaml'))['data_process']['split_ratio']

# import data

data_file_name = sys.argv[1]

full_data = pd.read_csv(data_file_name)

print('Null values for each feature:', full_data.isnull().sum())

full_data_cleaned = full_data.fillna(full_data.mean()) 
print('Sum of Null values for each feature post treatment:', sum(full_data_cleaned.isnull().sum()))
# train test split 
train, test = train_test_split(full_data_cleaned , test_size = split_ratio)

# create folder to save file
data_path = 'processed_data'
os.makedirs(data_path, exist_ok = True)

# saving prepared data
train.to_csv(os.path.join(data_path, 'out_train.csv'), index = False)
test.to_csv(os.path.join(data_path, 'out_test.csv'), index = False)