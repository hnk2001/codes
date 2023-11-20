import pandas as pd

df = pd.read_csv('emails.csv')
# reads the data file 

print(df.shape)
# prints rows x colums

print(df.head())
# prints the first 5 entries of dataset

#-------------------------------------------------
# STEP 1) :- seperation of input and output data.
#-------------------------------------------------

# INPUT DATA : - 
x = df.drop(['Email No.', 'Prediction'], axis = 1)

#OUTPUT DATA : - 
y = df['Prediction']

# printing the change in no. of columns
print(x.shape)

import seaborn as sns

# bar diagram representation of spam emails
sns.countplot(x = y)

print(y.value_counts())

#----------------------------------------------------------------------------
# STEP 2) :- 
# K-MEANS := IF ONE COLUMNS VALUE IS MORE, IT HAS MORE IMPACT ON THE OUTPUT.
# HENCE, TO BALANCE THIS OUT WE USE THE MIN-MAX SCALING.
#----------------------------------------------------------------------------

# taking care of scaling libraries
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# scaling the input data only
x_scaled = scaler.fit_transform(x)
print(x_scaled) 


#----------------------------------------------------
# STEP 3) :-
# CROSS-VALIDATION
# TRAINING AND TESTING DATA SPLIT INTO 75% AND 25%
#----------------------------------------------------

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, random_state=0, test_size=0.25)

print(x_train.shape, x_test.shape)


#--------------------------------------------------------
# STEP 4) :-
# NOW WE WILL USE K-NEAREST-NEIGHBOUR(KNN) ALGORITHM 
#--------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)

# TRAIN THE ALGORITHM MODEL
knn.fit(x_train, y_train)


# TESTING THE TRAINED MODEL
y_pred = knn.predict(x_test)

#------------------------------------------------
# STEP 5) :-
# NOW WE WILL TEST THE ACCURACY OF OUR MODEL
# USING Y_PRED AND Y_TEST AND CONFUSION MATRIX
#------------------------------------------------

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import classification_report

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))


#-------------------------------------------------------------------------------
# STEP 6) :-
# HERE WE TOOK RANDOM VALUE OF K 
# BUT TO FIND THE IDEAL K-VALUE THERE IS A METHOD, IT WILL INCREASE THE ACCURACY
#-------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

error = []
for k in range(1,41):
	knn = KNeighborsClassifier(n_neighbors = k)
	knn.fit(x_train, y_train)
	pred = knn.predict(x_test)
	error.append(np.mean(pred != y_test))
print(error)

#-------------------------------------------------------------------------------
# STEP 7) :-
# from calculating the error we get that the error is min when n_neighbors = 1 
# so we calculate using n_neighbors as 1
#-------------------------------------------------------------------------------


knn = KNeighborsClassifier(n_neighbors = 1)

# TRAIN THE ALGORITHM MODEL
knn.fit(x_train, y_train)


# TESTING THE TRAINED MODEL
y_pred = knn.predict(x_test)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))


#----------------------------------------------
# STEP 8) ;-
# USING SVM AND COMPARING THE ACCURACIES
#----------------------------------------------

from sklearn.svm import SVC

svm = SVC(kernel = "linear")

svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)

print(accuracy_score(y_test, y_pred))



































