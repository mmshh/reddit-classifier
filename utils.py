import csv
import numpy as np


def convert_to_csv(data):
    csv_columns = ['Id', 'Category']
    with open('./resources/output.csv', 'w') as f:
        writer = csv.DictWriter(f, csv_columns)
        writer.writeheader()
        writer.writerows(data)


def read_test_data():
    value = np.load("./resources/data_test.pkl", allow_pickle=True)
    return value


def read_train_data():
    return np.load("./resources/data_train.pkl", allow_pickle=True)
