#-------------------------------------------------------------------------
# AUTHOR: Harshitha Patnaik
# FILENAME: knn.py
# SPECIFICATION: Calculate Highest KNN accuracy
# FOR: CS 5990- Assignment #3
# TIME SPENT: 2 hrs
#-----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

#defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

#reading the training data
df_training = pd.read_csv('weather_training.csv')
df_training['Formatted Date'] = pd.to_datetime(df_training['Formatted Date'], utc=True)
df_training['Formatted Date'] = (df_training['Formatted Date'] - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta('1s')
X_training = np.array(df_training.iloc[:, :-1]).astype('float')
y_training_raw = np.array(df_training.iloc[:, -1]).astype('float')
# Discretize the training class labels
y_training = np.digitize(y_training_raw, classes)

#reading the test data
df_test = pd.read_csv('weather_test.csv')
df_test['Formatted Date'] = pd.to_datetime(df_test['Formatted Date'], utc=True)
df_test['Formatted Date'] = (df_test['Formatted Date'] - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta('1s')
X_test = np.array(df_test.iloc[:, :-1]).astype('float')
y_test_raw = np.array(df_test.iloc[:, -1]).astype('float')
# Discretize the test class labels
y_test = np.digitize(y_test_raw, classes)
#hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')

# Variable to keep track of the highest accuracy found
highest_accuracy = 0
best_params = {'k': None, 'p': None, 'weight': None}

#loop over the hyperparameter values (k, p, and w) ok KNN
#--> add your Python code here
for k in k_values:
    for p in p_values:
        for w in w_values:
            #fitting the knn to the data
            #--> add your Python code here

            clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
            clf = clf.fit(X_training, y_training)
            correct_predictions = 0
            total_predictions = len(y_test)

            #make the KNN prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously, use zip()
            #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
            #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
            #--> add your Python code here
            for x_test_sample, y_test_sample in zip(X_test, y_test_raw):
                predicted = clf.predict([x_test_sample])[0]
                predicted_value = classes[predicted]
                if abs(predicted_value - y_test_sample) <= 0.15 * y_test_sample:
                    correct_predictions += 1

            #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
            #with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
            #--> add your Python code here
            accuracy = correct_predictions / total_predictions
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                best_params['k'] = k
                best_params['p'] = p
                best_params['weight'] = w

                print(f"Highest KNN accuracy so far: {highest_accuracy}, Parameters: k={k}, p={p}, weight='{w}'")





