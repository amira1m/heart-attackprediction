import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


data =pd.read_csv("heart.csv")
a= data.iloc[:, 0:13]
b= data['output']
b=b.to_frame()


def null(a):
    a.dropna(axis=0, how='any', inplace=True)
    return a


def outlier_detection(feature):
    # Select the columns we want to work with


    # Calculate the median and interquartile range (IQR) of each selected column
    column_medians = feature.median()
    column_iqrs = feature.quantile(0.75) - feature.quantile(0.25)

    # Calculate the lower and upper bounds for outliers for each column
    lower_bounds = column_medians - 1.5 * column_iqrs
    upper_bounds = column_medians + 1.5 * column_iqrs

    # Replace any values in the selected columns that fall outside the bounds with the column median
    for col in feature:
        feature.loc[(feature[col] < lower_bounds[col]) | (feature[col] > upper_bounds[col]), col] = column_medians[col]

    return feature

def data_scaling(feature):
    feature['oldpeak'] = zscore(feature['oldpeak'])
    feature['thalachh'] = zscore(feature['thalachh'])
    feature['age'] = zscore(feature['age'])
    feature['trtbps'] = zscore(feature['trtbps'])
    feature['chol'] = zscore(feature['chol'])
    return feature


def feature_encoder(a):
    mms = MinMaxScaler()
    tips_ds_mms = mms.fit_transform(a)
    tips_ds_mms_df = pd.DataFrame(tips_ds_mms,
                                  columns=a.columns)
    tips_ds_mms_df.head()
    #print(tips_ds_mms_df.head())

    return tips_ds_mms_df
