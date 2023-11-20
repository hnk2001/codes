import pandas as pd 
import seaborn as sns

df = pd.read_csv('Churn_Modelling.csv')

# PRINTING THE SHAPE OF DATASET AND COLUMNS NAME
print(df.shape)
print(df.columns)

# PRINT-FIRST_5- ENTRIES
print(df.head())

# INPUT DATA
x = df[['CreditScore', 'Age', 'Tenure', 'Balance',
 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary' ]]

# OUTPUT DATA
y = df['Exited']

# Y-VALUES DISTRIBUTION
print(y.value_counts())

#-----------------------------------------------------------------
# STEP 5) :-
# RUN THE COMMAND == (pip3 install imbalanced-learn)
# TO OVERGROW OR UNDERGROW THE ENTRIES
#-----------------------------------------------------------------

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
x_res, y_res = ros.fit_resample(x, y)
print(y_res.value_counts())

#-----------------------------------------------------------------
# STEP 1) :-
# NORMALIZING THE DATA 
# LEARN THE DIFFERENCE BETWEEN STANDARD SCALER AND MIN-MAX SCALER
#-----------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_res)
print(x_scaled)

#-----------------------------------------------------------------
# STEP 2) :-
# SPLITTING THE DATA INTO TRAINING AND TESTING SET 
# THE 75% FOR THE TRAINING AND 25% TESTING
#-----------------------------------------------------------------

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_res, random_state = 0, test_size=0.25)


#-----------------------------------------------------------------
# STEP 3) :-
# USING THE ARTIFICIAL NEURAL NETWORK (ANN) 
# IMPORTING THE LIABRARIES OF MLPClassifier
#-----------------------------------------------------------------

from sklearn.neural_network import MLPClassifier

ann = MLPClassifier(hidden_layer_sizes=(100, 100, 100),
	random_state = 0, max_iter=100, activation='relu')

ann.fit(x_train, y_train)

y_pred = ann.predict(x_test)


#---------------------------------------------------------------------------
# STEP 4) :-
# USING THE CLASSIFICATION_REPORT, CONFUSION-MATRIX-DISPLAY, ACCURACY_SCORE 
#---------------------------------------------------------------------------

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


























































































