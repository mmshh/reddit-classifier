from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from utils import convert_to_csv, read_train_data, read_test_data


def train_predict_sgd(x_t, y_t, x_test, a):
    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='hinge', penalty='elasticnet',alpha=a, random_state=42,
                            fit_intercept = False,  tol=0.00001, early_stopping=True, 
                                         n_iter_no_change=400, max_iter=20)),
                   ])
    sgd.fit(x_t, y_t)
    y_pred = sgd.predict(x_test)
    return y_pred


def train_predict_svm(to, c, x_t, y_t, x_test):
    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', LinearSVC(penalty='l2', loss='hinge', dual=True, tol=to, C=c, 
                                      multi_class='ovr', fit_intercept=False, intercept_scaling=1, 
                                      verbose=0, random_state=None, max_iter=20)),
                   ])
    sgd.fit(x_t, y_t)
    y_pred = sgd.predict(x_test)
    return y_pred


def generate_results(res, classes_name):
    toOutput = []
    for i in range(len(res)):
        toOutput.append({'Id': i, 'Category': classes_name[res[i]]})
    convert_to_csv(toOutput)


if __name__ == "__main__":
    train_data = read_train_data()
    comment = train_data[0]
    result = train_data[1]
    test_data = read_test_data()

    lab = []
    classes_name, classes_count = np.unique(result, return_counts=True)
    for i in range(len(result)):
        lab.append(np.where(classes_name == result[i])[0][0])
    lab = np.asarray(lab)
    alpha = 1e-5
    maxIt = 1
    result1 = train_predict_sgd(comment, lab, test_data, alpha)
    result2 = train_predict_svm(alpha, maxIt, comment, lab, test_data)

    generate_results(result1, classes_name)
    # generate_results(result2, classes_name)