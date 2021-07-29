'''
#Log Odds
In Linear Regression we multiply the coefficients of our features by their respective feature values and add the intercept,
resulting in our prediction, which can range from -∞ to +∞. In Logistic Regression, we make the same multiplication of feature
coefficients and feature values and add the intercept, but instead of the prediction, we get what is called the log-odds.

The odds tell us how many more times likely an event is to occur than not occur. If a student will pass the exam with probability 0.7,
they will fail with probability 0.3.We can then calculate the odds of passing as:

                Odds of passing = 0.7/0.3=2.33

The log-odds are then understood as the logarithm of the odds!

                log(2.33) = 0.847
'''

import numpy as np
from exam import hours_studied, calculated_coefficients, intercept

def log_odds(features, coefficients, intercept):
    return np.dot(features, coefficients)+intercept  #np.dot is used to find dot product of two matrix

print(log_odds(hours_studied, calculated_coefficients, intercept))



'''
#Sigmoid Function
The Sigmoid Function is a special case of the more general Logistic Function, where Logistic Regression gets its name.
By plugging the log-odds into the Sigmoid Function, we map the log-odds to the range [0,1].

                h(z) = 1/(1+e^(-z))

e^(-z) can be written in numpy as np.exp(-z)
'''


import codecademylib3_seaborn
import numpy as np
from exam import calculated_log_odds

def sigmoid(z):
    denominator = 1+np.exp(-z)
    return 1/denominator

print(sigmoid(calculated_log_odds))



'''
#Classification Thresholding
The default threshold for many algorithms is 0.5. If the predicted probability of an observation belonging to the positive
class is greater than or equal to the threshold, 0.5, the classification of the sample is the positive class. If the predicted
probability of an observation belonging to the positive class is less than the threshold, 0.5, the classification of the sample
is the negative class.
'''

def predict_class(features, coefficients, intercept, threshold):
    calculated_log_odds = log_odds(features, coefficients, intercept)
    probabilities = sigmoid(calculated_log_odds)
    return np.where(probabilities >= threshold, 1, 0)



'''
#Using sklearn to create Logistic Regression Model

While usign sklearn's Logistic Regression model we need to keep in mind that it takes normalized features!
'''

import numpy as np
from sklearn.linear_model import LogisticRegression
from exam import hours_studied_scaled, passed_exam, exam_features_scaled_train,
    exam_features_scaled_test, passed_exam_2_train, passed_exam_2_test, guessed_hours_scaled

model = LogisticRegression()
model.fit(hours_studied_scaled, passed_exam)

intercept = model.intercept_
calculated_coefficients = model.coef_
print(intercept)
print(calculated_coefficients)

passed_predictions = model.predict_proba(guessed_hours_scaled)
print(passed_predictions)
