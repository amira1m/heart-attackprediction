import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
from preprocessing import *



data =pd.read_csv("heart.csv")
a= data.iloc[:, 0:13]
b= data['output']
b=b.to_frame()

X_train, X_test, Y_train, Y_test = train_test_split(a,b, test_size=0.2, random_state=195, shuffle=True)



def anova(a, b, k=4):
    # Perform ANOVA feature selection with available samples
    f_values, p_values = f_classif(a, b)
    sorted_idx = np.argsort(f_values)[::-1]
    selected_features = a.iloc[:, sorted_idx[:k]]

    # Print the selected features
    print(selected_features.columns)
    return selected_features

