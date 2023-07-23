import sys
import os
import pickle
import json
import numpy as np
import pandas as pd
from dvclive import Live
from matplotlib import pyplot as plt

from sklearn.metrics import recall_score, precision_score, f1_score , accuracy_score, roc_auc_score

model_path = sys.argv[1]
with open(model_path, 'rb') as f:
    model = pickle.load(f)

test_file_name = sys.argv[2]
test_data = pd.read_csv(test_file_name)
test = test_data.drop('Churn', axis = 1)
labels = test_data['Churn']

predictions_proba = model.predict_proba(test)
test_predictions = model.predict(test)

acc = accuracy_score(labels, test_predictions)
prec = precision_score(labels, test_predictions)
rec = recall_score(labels, test_predictions)
f1 = f1_score(labels, test_predictions)

auc_score = roc_auc_score(labels, predictions_proba[:, 1])

EVAL_PATH = 'results'

os.makedirs(EVAL_PATH, exist_ok = True)

output = pd.DataFrame({ 'Actual' : labels, 'Predicted': test_predictions })
output.to_csv(os.path.join(EVAL_PATH , 'predictions_vs_actuals.csv'))

fig, axes = plt.subplots()
fig.subplots_adjust(left = 0.2, bottom = 0.2, top = 0.95)
importances = model.feature_importances_
indices = np.argsort(importances)
features = test.columns
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color = 'b', align = 'center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig(os.path.join(EVAL_PATH, 'feature_importance.png'))


with Live(os.path.join(EVAL_PATH, 'live'), dvcyaml=False) as live:
   
    live.log_metric('Test_accuracy_score', acc)
    live.log_metric('Test_precision_score', prec)
    live.log_metric('Test_recall_score', rec)
    live.log_metric('Test_f1_score', f1)
    live.log_metric('AUC_score', auc_score)

    live.log_sklearn_plot('roc', labels.squeeze(), predictions_proba[:, 1], name = 'ROC Curve')
    live.log_sklearn_plot('confusion_matrix',
                          labels.squeeze(),
                          test_predictions,
                          name = 'Confusion-matrix'
                         )
    live.log_sklearn_plot('precision_recall', 
                      labels.squeeze(), 
                      predictions_proba[:, 1],
                      name = 'Precision-Recall Curve'
                         )
    live.log_image('Feature_importance.png', os.path.join(EVAL_PATH, 'feature_importance.png'))
