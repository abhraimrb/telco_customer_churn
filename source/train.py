import os
import pickle
import sys
import pandas as pd
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier


#parameters
n_estimators = yaml.safe_load(open('params.yaml'))['train']['n_estimators']
max_depth = yaml.safe_load(open('params.yaml'))['train']['max_depth']
min_samples_split = yaml.safe_load(open('params.yaml'))['train']['min_samples_split']
min_samples_leaf = yaml.safe_load(open('params.yaml'))['train']['min_samples_leaf']
class_weight = yaml.safe_load(open('params.yaml'))['train']['class_weight']

#load train data
data_file_name = sys.argv[1]
train_data = pd.read_csv(data_file_name)

# model
model_path = sys.argv[2]

x_train = train_data.drop(columns = 'Churn')
y_train = train_data['Churn']

# model training 
model = RandomForestClassifier(n_estimators = n_estimators,max_depth = max_depth, 
                               min_samples_split = min_samples_split,
                               min_samples_leaf = min_samples_leaf, 
                               class_weight = class_weight )


model.fit(x_train, y_train)

os.makedirs('model_dir', exist_ok = True)

model_path = os.path.join('model_dir', 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
