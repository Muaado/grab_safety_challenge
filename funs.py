#Import libaries
from sklearn.externals import joblib
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix
from matplotlib import pyplot
import seaborn as sns


#Read the CSV Files
def read_holdout_data(relative_path):
    def read_data(file):  
        #import the datasets and labels
        d = pd.read_csv(file, delimiter=",", engine='python', header='infer', dtype={ 'bookingID': np.int64, 'Accuracy': np.int32, 'Bearing': np.int32, 'acceleration_x': np.float64, 'acceleration_y': np.float64, 'acceleration_z': np.float64, 'gyro_x': np.float64, 'gyro_y': np.float64, 'gyro_z': np.float64, 'second': np.int32, 'Speed': np.int32  })
        print(f"{file} : READ SUCCESSFULL")
        return d
    
    #fetch all files inside the features dir
    files = os.listdir(os.getcwd()+relative_path)

    #create and append individual csv files to one single data frame
    dataset = pd.DataFrame()
    for file in files:
        file = os.getcwd()+relative_path+file
        if file.endswith('.csv'):
            dataset = dataset.append(read_data(file))
    return dataset

# Read labels data
def read_labels(path):
    labels = pd.read_csv(os.getcwd()+path+'label.csv', delimiter=",", header='infer')
    #drop rows with BookingId values as 0
    labels = labels[labels.bookingID != 0]
    return labels


#Preprocess dataset    
def preprocess_holdout(dataset):
    def remove_outlier(df, column):
        quantile_one = df[column].quantile(0.25)
        quantile_three = df[column].quantile(0.75)
        IQR = quantile_three-quantile_one
        lower_fence  = quantile_one-1.5*IQR
        upper_fence = quantile_three+1.5*IQR
        df_out = df.loc[(df[column] > lower_fence) & (df[column] < upper_fence)]
        return df_out
    
    pd.options.display.float_format = '{:.2f}'.format

    #drop rows with BookingId values as 0
    dataset = dataset[dataset.bookingID != 0]
    dataset = remove_outlier(dataset, 'second')
    dataset = dataset.sort_values(['bookingID','second'],ascending=True)
    return dataset


#Preprocess labels    
def preprocess_labels(labels):
    labels = labels[labels.bookingID != 0]
    labels.drop_duplicates(subset ="bookingID", keep = False, inplace = True) 
    return labels