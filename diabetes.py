import pandas as pd
import seaborn as sns

df = pd.read_csv('diabetes.csv')
print(df.head())
print(df.columns)

#-----------------------------------------------
# STEP 1) :-
# SEGREGATING THE INPUT AND OUTPUT
#-----------------------------------------------

#INPUT DATA
x = df.drop('Outcome', axis = 1)

#OUTPUT DATA
y = df['Outcome']

sns.countplot(x = y)

print(y.value_counts())

#---------------------------------------------------------------------------------
# STEP 2) :-
# SCALING THE MIN-MAX-SCALER 
# TO BALANCE OUT THE ENTRIES AND REMOVE THE OUTCOME CHANGE DUE TO OVERDEPENDENCE
#---------------------------------------------------------------------------------

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_scaled = scaler.fit_transform(x)

#---------------------------------------------------------------------------------
# STEP 3) :-
# SPLITTING OF TEST AND TRAIN DATA INTO 75% AND 25%
#---------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, random_state=0, test_size=0.25)


#---------------------------------------------------------------------------------
# STEP 4) :-
# USING THE K-NEAREST-NEIGHBORS AALGORITHM(KNN)
#---------------------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

#---------------------------------------------------------------------------------
# STEP 5) :-
# CHECKING THE ACCURACY_SCORE, CONFUSION-MATRIX-DISPLAY, CLASSIFICATION-REPORT
#---------------------------------------------------------------------------------

from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
print(classification_report(y_test, y_pred))

#---------------------------------------------------------------------------------
# STEP 6) :-
# FINDING THE IDEAL VALUE OF N_NEIGHBORS
#---------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

error = []
for k in range(1,41):
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(x_train, y_train)
	pred = knn.predict(x_test)
	error.append(np.mean(pred != y_test))

print(error)

plt.figure(figsize=(16,9))
plt.xlabel('Value of K') 
plt.ylabel('Error')
plt.grid()
plt.xticks(range(1,41))
plt.plot(range(1,41), error, marker='.')

#---------------------------------------------------------------------------------
# STEP 7) :-
# FOUND THE IDEAL VALUE OF N_NEIGHBORS AT 33
#---------------------------------------------------------------------------------


knn = KNeighborsClassifier(n_neighbors=33)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print(classification_report(y_test, y_pred))












































































