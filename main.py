import numpy as np

from data_preparation import load_data
from model_training import train_model
from model_evaluation import evaluate_model


if __name__ == '__main__':
    train_file = './dataset/UNSW-NB15_1.csv'
    test_file = './dataset/UNSW-NB15_2.csv'

    X_train, y_train, X_test, y_test = load_data(train_file, test_file)

    num_classes = len(np.unique(y_train))

    model = train_model(X_train, y_train, num_classes)

    evaluate_model(model, X_test, y_test)
