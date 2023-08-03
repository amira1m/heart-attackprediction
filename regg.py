from preprocessing import *
from feature_selection import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import metrics
import time
import joblib
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression

data =pd.read_csv("heart.csv")
a= data.iloc[:, 0:13]
b= data['output']
b=b.to_frame()

X_train, X_test, Y_train, Y_test = train_test_split(a,b, test_size=0.2, random_state=195, shuffle=True)

X_train=null(X_train)
X_train=outlier_detection(X_train)
X_train=feature_encoder(X_train)


X_test=null(X_test)
X_test=outlier_detection(X_test)
X_test=feature_encoder(X_test)

#print(X_train)
selected_features = anova(X_train, Y_train)
X_train = selected_features
X_test = X_test[selected_features.columns]

print("1: Multiple regression model\n2: Polynomial regression model")
choice = int(input("Choose your model: "))

# multiple regression model
if choice == 0:



    # Apply Linear Regression on the selected features
    cls = linear_model.LinearRegression()
    startTrain = time.time()
    cls.fit(X_train, Y_train)
    endTrain = time.time()
    start_test = time.time()
    prediction = cls.predict(X_test)
    end_test = time.time()
    print('Co-efficient of multiple regression', cls.coef_)
    print('Intercept of multiple regression model', cls.intercept_)
    print('R2 Score', metrics.r2_score(Y_test, prediction))
    print('Mean Square Error', metrics.mean_squared_error(np.asarray(Y_test), prediction))
    print("Actual time for training", endTrain - startTrain)
    print("Actual time for Testing", end_test - start_test)
    with open(' multiple  regression', 'wb') as filename:
         pickle.dump(cls, filename)

# polynomial regression model
elif choice == 1:
    poly_features = PolynomialFeatures(degree=2)
    # transform existing features to higher degree features
    X_train_poly = poly_features.fit_transform(X_train)

    # fit the transformed features to Linear Regression and calculate time for training
    poly_model = linear_model.LinearRegression()
    startTrain = time.time()
    poly_model.fit(X_train_poly, Y_train)
    endTrain = time.time()

    # predicting on training data-set and calculate time for testing
    y_train_predicted = poly_model.predict(X_train_poly)
    start_test = time.time()
    Y_pred = poly_model.predict(poly_features.transform(X_test))
    end_test = time.time()

    # predicting on test data-set
    prediction = poly_model.predict(poly_features.fit_transform(X_test))

    # print Co-efficient and statistics for polynomial regression
    print('Co-efficient of linear regression', poly_model.coef_)
    print('Intercept of linear regression model', poly_model.intercept_)
    print('R2 Score', metrics.r2_score(Y_test, prediction))
    print('Mean Square Error', metrics.mean_squared_error(Y_test, prediction))
    print("Actual time for training", endTrain - startTrain)
    print("Actual time for Testing", end_test - start_test)

    # test polynomial model on first sample
    true_pofit_value = np.asarray(Y_test)[0]
    predicted_profit_value = prediction[0]
   # print("The true profit value " + str(true_price_value))
    #print("The predicted profit  value " + str(predicted_price_value))

    joblib.dump(poly_model, 'polymodel')
    joblib.dump(poly_features, 'poilynomial_features_model')

