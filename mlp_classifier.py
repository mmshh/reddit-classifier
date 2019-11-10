import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

from utils import read_test_data, convert_to_csv, read_train_data


def cross_validate(clf, X_train, y_train, cv):
    """
    Helper method to cross validate
    :param clf: classifier
    :param X_train:
    :param y_train:
    :param cv:
    :return: void, prints the accuracies
    """
    accuracies = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=cv)
    print(accuracies)


def generate_results(res, classes_name):
    """
    Converts the predictions to csv and creates the 'output.csv' file in the resources folder
    :param res:
    :param classes_name:
    :return: void
    """
    toOutput = []
    for i in range(len(res)):
        toOutput.append({'Id': i, 'Category': classes_name[res[i]]})
    convert_to_csv(toOutput)


def fit_predict(X_train, label_nums, test_data):
    """
    Fits and makes the class predictions using a MLP classifier
    :param X_train:
    :param label_nums: The labels numbered
    :param test_data: Data for which we want to make predictions
    :return: void
    """
    clf = MLPClassifier(verbose=True, early_stopping=True)
    clf.fit(X_train, label_nums)
    pred = clf.predict(vectorizer.transform(test_data))
    generate_results(pred, classes_name)


if __name__ == "__main__":
    train_data = read_train_data()
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(train_data[0])
    X_train = vectorizer.transform(train_data[0])

    test_data = read_test_data()
    label_numbers = []
    result = train_data[1]
    classes_name, classes_count = np.unique(result, return_counts=True)
    for i in range(len(result)):
        label_numbers.append(np.where(classes_name == result[i])[0][0])
    label_numbers = np.asarray(label_numbers)

    fit_predict(X_train, label_numbers, test_data)
    # cross_validate(clf, X_train, result, 4)
