import pickle
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
# Assuming you have the pre_processing module and its functions available
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import  time
from preprocessing import *
from feature_selection import *


data =pd.read_csv("heart.csv")
a= data.iloc[:, 0:13]
b= data['output']
b=b.to_frame()

X_train, X_test, Y_train, Y_test = train_test_split(a,b, test_size=0.2, random_state=195, shuffle=True)

X_test.fillna(0, inplace=True)
Y_test.fillna(0, inplace=True)

X_train=null(X_train)
X_train=outlier_detection(X_train)
#X_train=feature_encoder(X_train)
X_train=data_scaling(X_train)

X_test=null(X_test)
X_test=outlier_detection(X_test)
#X_test=feature_encoder(X_test)
X_test = data_scaling(X_test)



#print(X_train)
selected_features = anova(X_train, Y_train)
X_train = selected_features
X_test = X_test[selected_features.columns]


#print(X_train)

def random_forest(X_train, Y_train, X_test, Y_test):
    rfc = RandomForestClassifier(random_state=42)

    # Define the hyperparameter grid to search over
    param_grid = {
        'n_estimators': [60,160,70],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2']
    }

    # Create a grid search object and fit it to the training data
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1)
    startTrain = time.time()
    grid_search.fit(X_train, Y_train.values.ravel())

    # Print the best hyperparameters and corresponding accuracy
    print("Best parameters:", grid_search.best_params_)
    print("Best accuracy:", grid_search.best_score_)
    endTrain = time.time()
    # Test the Random Forest classifier on the testing data using the best hyperparametersh
    best_rfc = grid_search.best_estimator_
    start_test = time.time()
    Y_test_pred = best_rfc.predict(X_test)
    end_test = time.time()
    print('random tree test data accuracy: ', accuracy_score(Y_test, Y_test_pred))
    print("Actual time for training", endTrain - startTrain)
    print("Actual time for Testing", end_test - start_test)

    model1 = best_rfc
    with open('random_forest.pkl', 'wb') as filename:
         pickle.dump(model1, filename)




# logistic model
def logistic(X_train, Y_train, X_test, Y_test):
    clf = LogisticRegression(random_state=195, solver='newton-cg', penalty='none', max_iter=1000)
    startTrain = time.time()
    clf.fit(X_train, Y_train)

    y_train_pred = clf.predict(X_train)
    endTrain = time.time()

    start_test = time.time()
    pred=clf.predict(X_test)
    end_test = time.time()
    print(" the train accuracy of logistic model equals:", accuracy_score(Y_train, y_train_pred))
    acc = accuracy_score(Y_test, pred)
    print("the test accuracy of logistic model equals :", acc)
    print("Actual time for training", endTrain - startTrain)
    print("Actual time for Testing", end_test - start_test)
    model2=clf
    with open('logistic_regressionModel.pkl', 'wb') as filename:
         pickle.dump(model2, filename)

def SVM_rbf( X_train, Y_train, X_test, Y_test):
        # SVM classification
        clf = svm.SVC(kernel='rbf', gamma=0.5, C=10)  # rbf Kernel
        startTrain = time.time()
        clf.fit(X_train, Y_train)
        y_train_pred = clf.predict(X_train)
        endTrain = time.time()
        rbf_train_time = endTrain-startTrain
        print(" svm_rbf train data Accuracy:", accuracy_score(Y_train, y_train_pred))
        start_test = time.time()
        prediction = clf.predict(X_test)
        end_test = time.time()
        rbf_train_time=end_test-start_test
        print("svm_rbf test data Accuracy:", metrics.accuracy_score(Y_test, prediction))
        print(classification_report(Y_test, prediction))
        print("Actual time for training", endTrain - startTrain)
        print("Actual time for Testing", end_test - start_test)
        model3 = clf
        with open('rbf_Model.pkl', 'wb') as filename:
             pickle.dump(model3, filename)


def SVM_poly(X_train, Y_train, X_test, Y_test):
    # SVM classification
    clf = svm.SVC(kernel='poly', degree=2, C=100)  # poly Kernel
    startTrain = time.time()
    clf.fit(X_train, Y_train)
    y_train_pred = clf.predict(X_train)
    endTrain = time.time()
    print(" svm_poly train data Accuracy:", accuracy_score(Y_train, y_train_pred))
    start_test = time.time()
    prediction = clf.predict(X_test)
    end_test = time.time()
    print(" svm_poly test data Accuracy:", metrics.accuracy_score(Y_test, prediction))
    print(classification_report(Y_test, prediction))
    print("Actual time for training", endTrain - startTrain)
    print("Actual time for Testing", end_test - start_test)
    model4 = clf
    with open('poly_Model.pkl', 'wb') as filename:
         pickle.dump(model4, filename)
    #return model4



#SVM_rbf(X_train, Y_train, X_test, Y_test)
#SVM_poly(X_train, Y_train, X_test, Y_test)
#random_forest(X_train, Y_train, X_test, Y_test)
#logistic(X_train, Y_train, X_test, Y_test)
#SVM(X_train, Y_train, X_test, Y_test)
