import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import torch

def load_data(train_file, test_file):
    # Load train and test data
    train_df = pd.read_csv(train_file, header=None, dtype=str)
    test_df = pd.read_csv(test_file, header=None, dtype=str)
    
    # # check if there are any samples in the dataframe
    # if train_df.empty:
    #     raise ValueError("Training dataframe is empty")
    
    # # check if there are any samples in the dataframe
    # if test_df.empty:
    #     raise ValueError("Testing dataframe is empty")
    
    # Check for non-numeric values in the training dataframe
    train_df_numeric = train_df.apply(pd.to_numeric, errors='coerce')
    # if train_df_numeric.isna().all().all():
    #     raise ValueError("No numeric samples in the training dataframe")
    

    # Combine train and test data for label encoding
    combined_df = pd.concat([train_df, test_df], axis=0)
    
    # Drop rows with non-numeric values
    combined_df = combined_df.apply(pd.to_numeric, errors='coerce').dropna()
    train_df = train_df.loc[combined_df.index]
    test_df = test_df.loc[combined_df.index]
    
    # # check if there are any samples in the dataframe after dropping non-numeric rows
    # if train_df.empty:
    #     raise ValueError("No numeric samples in the training dataframe")    
    # elif train_df_numeric.isna().all().all():
    #     raise ValueError("No numeric samples in the training dataframe after dropping non-numeric rows")
    

    # Label encode categorical features
    categorical_features = [1, 2, 3, 41]
    for feature in categorical_features:
        encoder = LabelEncoder()
        encoder.fit(combined_df[feature].astype(str))
        train_df[feature] = encoder.transform(train_df[feature])
        test_df[feature] = encoder.transform(test_df[feature])

    # Scale numerical features
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    numerical_features = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    
    if not train_df.empty:
        train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
    if not test_df.empty:
        test_df[numerical_features] = scaler.transform(test_df[numerical_features])
    # train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
    # test_df[numerical_features] = scaler.transform(test_df[numerical_features])

    # Split dataset into features and labels
    X_train = train_df.drop([41, 42], axis=1).values
    y_train = train_df[41].values
    X_test = test_df.drop([41, 42], axis=1).values
    y_test = test_df[41].values

    # Convert data to PyTorch tensors
    # X_train = torch.tensor(X_train, dtype=torch.float32)
    X_train = torch.tensor(X_train.astype('float32'))
    y_train = torch.tensor(y_train, dtype=torch.long)
    # X_test = torch.tensor(X_test, dtype=torch.float32)
    X_test = torch.tensor(X_test.astype('float32'))
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, y_train, X_test, y_test


