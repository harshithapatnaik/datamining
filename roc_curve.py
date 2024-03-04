# -------------------------------------------------------------------------
# AUTHOR: Harshitha Patnaik
# FILENAME: roc_curve.py
# SPECIFICATION: draws ROC curve for a decision tree classifier
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 2 hours
# -----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd

# read the dataset cheat_data.csv and prepare the data_training numpy array
# --> add your Python code here
data = pd.read_csv('cheat_data.csv')

# Preprocess the dataset
# Remove 'k' from 'Taxable Income' and convert to float
data['Taxable Income'] = data['Taxable Income'].str.replace('k', '').astype(float)

# One-hot-encode 'Marital Status'
data = pd.get_dummies(data, columns=['Marital Status'])

# Convert 'Refund' to binary
data['Refund'] = data['Refund'].map({'Yes': 1, 'No': 0})

# Convert 'Cheat' to binary
data['Cheat'] = data['Cheat'].map({'Yes': 1, 'No': 0})

X = data.drop('Cheat', axis=1)
y = data['Cheat']

# split into train/test sets using 30% for test
trainX, testX, trainy, testy = train_test_split(X, y, test_size = 0.3) #, random_state=42)

# generate a no skill prediction (random classifier - scores should be all zero)
ns_probs = [0 for _ in range(len(testy))]

# fit a decision tree model by using entropy with max depth = 2
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=2)
clf.fit(trainX, trainy)

# predict probabilities for all test samples (scores)
# dt_probs = clf.predict_proba(testX)

# keep probabilities for the positive outcome only
dt_probs = clf.predict_proba(testX)[:, 1]

# calculate scores by using both classifiers (no skilled and decision tree)
ns_auc = roc_auc_score(testy, ns_probs)
dt_auc = roc_auc_score(testy, dt_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (dt_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()

# show the plot
pyplot.show()