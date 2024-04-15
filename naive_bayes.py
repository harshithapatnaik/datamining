#-------------------------------------------------------------------------
# AUTHOR: Harshitha Patnaik
# FILENAME: naive_bayes.py
# SPECIFICATION: Naive Bayes accuracy calculation
# FOR: CS 5990- Assignment #3
# TIME SPENT: 1 hrs
#-----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

#reading the training data
#--> add your Python code here
df_training = pd.read_csv('weather_training.csv')
df_training['Formatted Date'] = pd.to_datetime(df_training['Formatted Date'], utc=True)
df_training['Formatted Date'] = (df_training['Formatted Date'] - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta('1s')
X_training = np.array(df_training.iloc[:, :-1]).astype('float')
y_training_raw = np.array(df_training.iloc[:, -1]).astype('float')

#update the training class values according to the discretization (11 values only)
#--> add your Python code here
y_training = np.digitize(y_training_raw, classes)

#reading the test data
#--> add your Python code here
df_test = pd.read_csv('weather_test.csv')
df_test['Formatted Date'] = pd.to_datetime(df_test['Formatted Date'], utc=True)
df_test['Formatted Date'] = (df_test['Formatted Date'] - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta('1s')
X_test = np.array(df_test.iloc[:, :-1]).astype('float')
y_test_raw = np.array(df_test.iloc[:, -1]).astype('float')

#update the test class values according to the discretization (11 values only)
#--> add your Python code here
y_test = np.digitize(y_test_raw, classes)

#fitting the naive_bayes to the data
clf = GaussianNB()
clf = clf.fit(X_training, y_training)
correct_predictions = 0

#make the naive_bayes prediction for each test sample and start computing its accuracy
#the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
#to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
#--> add your Python code here
for x_test_sample, y_test_sample_real in zip(X_test, y_test_raw):
    predicted_class = clf.predict([x_test_sample])[0]
    predicted_value = classes[predicted_class]

    # calculate the percentage difference
    percentage_difference = 100 * abs(predicted_value - y_test_sample_real) / y_test_sample_real

    # if the prediction is within the correct range, increment the correct_predictions count
    if percentage_difference <= 15:
        correct_predictions += 1

# calculate the accuracy
accuracy = correct_predictions / len(X_test)

#print the naive_bayes accuracyy
#--> add your Python code here
print("naive_bayes accuracy: " + str(accuracy))



