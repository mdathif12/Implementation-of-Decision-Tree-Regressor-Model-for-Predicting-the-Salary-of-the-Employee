# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.calculate Mean square error,data prediction and r2. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: mohamed athif rahuman J
RegisterNumber:  212223220058
*/
import pandas as pd
data=pd.read_csv("C:/Users/admin/Documents/Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Position']=le.fit_transform(data['Position'])
data.head()

x=data[['Position','Level']]
y=data['Salary']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_predict)
mse

r2=metrics.r2_score(y_test,y_predict)
r2

dt.predict([[5,6]])
```

## Output:

![image](https://github.com/mdathif12/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149365313/2cd2573a-461a-40ef-8fd2-3da192e1985b)

![image](https://github.com/mdathif12/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149365313/3f0e69b1-6275-43d3-9f29-4176c65b24cf)

![image](https://github.com/mdathif12/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149365313/9ddc272f-37f8-400a-a51d-6184a4e0ba08)

![image](https://github.com/mdathif12/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149365313/63be1bb6-d7cc-4aa2-8ac2-f980e3a1e1c8)

![image](https://github.com/mdathif12/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149365313/6fee00f0-8349-4d76-94a2-b44042d3eff6)

![image](https://github.com/mdathif12/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149365313/eefd1488-0e99-46dc-9ec2-1e89c2fbfa89)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
