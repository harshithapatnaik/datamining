# -------------------------------------------------------------------------
# AUTHOR: Harshitha Patnaik
# FILENAME: decision_tree.py
# SPECIFICATION: reads two test dataset files to build a corresponding decision tree, averaging the accuracies
#               as the final classification performance of each model
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 8-10 hours
# -----------------------------------------------------------*/


# Importing necessary libraries
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Initializing paths for datasets
dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv']
test_dataset_path = 'cheat_test.csv'


# Function to preprocess data
def preprocess_data(df):
    # One-hot encode 'Marital Status'
    df = pd.get_dummies(df, columns=['Marital Status'])

    # Handling 'Taxable Income' with 'k', converting to float
    df['Taxable Income'] = df['Taxable Income'].str.replace('k', '').astype(float) * 1000

    # Encode 'Refund' and 'Cheat' columns
    df['Refund'] = df['Refund'].map({'Yes': 1, 'No': 0})
    df['Cheat'] = df['Cheat'].map({'Yes': 1, 'No': 0})

    return df


# Loading and preprocessing the test dataset
test_df = pd.read_csv(test_dataset_path)
test_df = preprocess_data(test_df)
X_test = test_df.drop(['Cheat'], axis=1).values
Y_test = test_df['Cheat'].values

# Main loop for training and testing
for ds in dataSets:
    accuracies = []  # to store accuracies for each run
    df_train = pd.read_csv(ds)
    df_train = preprocess_data(df_train)
    X_train = df_train.drop(['Cheat'], axis=1).values
    Y_train = df_train['Cheat'].values

    for i in range(10):
        # Training the decision tree model
        clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=None)
        clf.fit(X_train, Y_train)

        # Testing the model
        predictions = clf.predict(X_test)
        accuracy = np.mean(predictions == Y_test)
        accuracies.append(accuracy)

        # plotting the tree at data[3]
        if i == 3:
            plt.figure(figsize=(8, 4))
            tree.plot_tree(clf, feature_names=df_train.drop(['Cheat'], axis=1).columns, class_names=['No', 'Yes'],
                           filled=True, rounded=True)
            plt.show()

    # Calculating the average accuracy
    average_accuracy = np.mean(accuracies)
    print(f"Final accuracy when training on {ds}: {average_accuracy:.2f}")

