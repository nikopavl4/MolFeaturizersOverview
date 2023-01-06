import deepchem as dc
import numpy as np
from ml.config import experiment_args
import random
from ml.utils.models import create_model
from ml.utils.preprocessing import load_dataset, preprocessing_pipeline
import time

# Get arguments for the specific experiment
args = experiment_args()
print(f"Script arguments: {args}\n")

# ensure reproducibility
random.seed(args.seed)
np.random.seed(args.seed)

df, tasks = load_dataset(args)

dataset, transformer = preprocessing_pipeline(df, tasks, args)

#Data Splitting
splitter = dc.splits.RandomSplitter()
train_dataset, test_dataset = splitter.train_test_split(dataset)

# Define our evaluation metric which in our case is ROC AUC score
metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

# Create model defined in args
model = create_model(tasks,args)

# Fit trained model
start = time.time()
if(args.featurizer.lower() == 'convmol' or args.featurizer.lower() == 'weave'):
    model.fit(train_dataset, nb_epoch=10)
else:
    model.fit(train_dataset)
stop = time.time()
training_time = stop - start
print(f"Training time: {training_time} sec")

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], [transformer])
test_scores = model.evaluate(test_dataset, [metric], [transformer])

print("Train scores")
print(train_scores)

print("Test scores")
print(test_scores)

import csv
  
# Save Results to csv file
results = [train_scores['roc_auc_score'], test_scores['roc_auc_score'], training_time]
import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, f'Results/{args.dataset}/{args.model}/{args.seed}_{args.featurizer}.csv')
# Example.csv gets created in the current working directory
with open (filename,'w',newline = '') as csvfile:
    my_writer = csv.writer(csvfile, delimiter = ',')
    my_writer.writerow(results)


