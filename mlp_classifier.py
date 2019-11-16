import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
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


def preprocess(s):
    """
    Preprocessor to stem words
    :param s:
    :return: stemmed words
    """
    ps = nltk.PorterStemmer()
    return ps.stem(s)


def grid_search(mlp, X_train, label_numbers):
    """
    Performs a grid search to find the best parameters
    :param mlp: model
    :param X_train: train data
    :param label_numbers: labels in numbered format
    :return: void, prints the best parameters
    """
    parameter_space = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    clf.fit(X_train, label_numbers)
    print('Best parameters found:', clf.best_params_)


def fit_predict(X_train, label_nums, test_data, classes_name, vectorizer):
    """
     Fits and makes the class predictions using a MLP classifier
    :param X_train:
    :param label_nums: The labels numbered
    :param test_data: Data for which we want to make predictions
    :param classes_name:
    :param vectorizer:
    :return: predictions
    """
    clf = MLPClassifier(verbose=True, early_stopping=True, activation='tanh', learning_rate='adaptive')
    clf.fit(X_train, label_nums)
    pred = clf.predict(vectorizer.transform(test_data))
    generate_results(pred, classes_name)
    return pred


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

    predictions = fit_predict(X_train, label_numbers, test_data, classes_name, vectorizer)
    # cross_validate(MLPClassifier(verbose=True, early_stopping=True), X_train, result, 3)
    # grid_search(MLPClassifier(verbose=True, early_stopping=True), X_train, label_numbers)