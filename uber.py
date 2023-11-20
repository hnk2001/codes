import pandas as pd
import matplotlib.pyplot as plt
import seaborn  as sns
import datetime as dt
import warnings
import numpy as np

warnings.filterwarnings('ignore')

df = pd.read_csv("uber.csv")

df.head()
df.describe()
df.dropna(inplace=True)
df.isnull().sum()
df.drop(columns=['Unnamed: 0', 'key'], axis=1)

df['week_day'] = df['pickup_datetime'].dt.day_name()
df['year']=df['pickup_datetime'].dt.year
df['month']=df['pickup_datetime'].dt.month
df['hour']=df['pickup_datetime'].dt.hour
df.drop(['pickup_datetime'],axis=1)


def remove_outliers(df1, col):
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker = Q1 - 1.5*IQR
    upper_whisker = Q3 + 1.5*IQR
    print("#------------------------#")
    print("col = ", col, "Q1 = ", Q1, "Q3 = ", Q3)
    print("#------------------------#")
    df1[col] = np.clip(df1[col], lower_whisker, upper_whisker)
    return df1


def treat_all_outliers(df1, col_list):
    print("col_list", col_list)
    for c in col_list:
        df1 = remove_outliers(df1, c)
    return df1

if __name__ == "__main__":
    treat_all_outliers(df, df.columns)


corrMatrix = df.corr()
sns.heatmap(corrMatrix,annot=True)
plt.show()


from sklearn.model_selection import train_test_split

x= df.drop(['fare_amount'],axis=1)
y = df['fare_amount']

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.25)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
lr_pred = lr.predict(x_test)


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=100,random_state=101)
rfr.fit(x_train,y_train)
rfr_pred = rfr.predict(x_test)

from sklearn.metrics import mean_squared_error
lrmodel_rmse = np.sqrt(mean_squared_error(lr_pred,y_test))
print("mean squared error for lr:= ",lrmodel_rmse)

rfr_rmse = np.sqrt(mean_squared_error(rfr_pred, y_test))
print("mean squared error for rfr:= ",rfr_rmse )


