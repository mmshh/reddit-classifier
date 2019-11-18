from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from utils import convert_to_csv, read_train_data, read_test_data

def get_datas():
    train_data = read_train_data()
    comment = train_data[0]
    result = train_data[1]
    test_data = read_test_data()

    lab = []
    classes_name, classes_count = np.unique(result, return_counts=True)
    for i in range(len(result)):
        lab.append(np.where(classes_name == result[i])[0][0])
    lab = np.asarray(lab)
    
    return comment, lab, test_data, classes_name

def cross_validate(to, c, X_train, y_train, cv):
    clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', LinearSVC(penalty='l2', loss='hinge', dual=True, tol=to, C=c, 
                                      multi_class='ovr', fit_intercept=False, intercept_scaling=1, 
                                      verbose=0, random_state=None, max_iter=20)),
                   ])
    
    accuracies = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=cv)
    print(accuracies)
    

def train_predict_svc(to, c, x_t, y_t, x_test):
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
    to = 0.0001
    c = 1
    comment, lab, test_data, classes_name = get_datas()
    
    result = train_predict_svc(to, c, comment, lab, test_data)

    generate_results(result, classes_name)
    #cross_validate(to, c, comment, lab, 4)