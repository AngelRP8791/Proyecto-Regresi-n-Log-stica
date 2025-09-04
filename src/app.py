# your code here
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

url  = "https://raw.githubusercontent.com/it-ces/Datasets/refs/heads/main/diabetes_prediction_dataset.csv"

df = pd.read_csv(url)

df['diabetes'].value_counts()

df.columns

X = ['age', 'bmi', 'blood_glucose_level','HbA1c_level']
target = 'diabetes'

X = df[X]
y = df[target]

def logist_cv(X_train, y_train):
  estimator = LogisticRegression()
  params  ={'max_iter': [100, 200,500],
            'C': [0.01, 0.1, 10, 100],
            'penalty':['l1', 'l2', 'elasticnet'],
            'tol' : [1e-06, 1e-05]}
  cv = KFold(n_splits=5, shuffle=True, random_state=123) # replicables...
  grid_search = GridSearchCV(estimator=estimator,
                                param_grid=params,
                                cv=cv,
                                scoring='accuracy')
  grid_result = grid_search.fit(X_train, y_train)
  return grid_search.best_estimator_

model = logist_cv(X_train, y_train)

preds = model.predict(X_test)

print(classification_report(y_test, preds))

model
