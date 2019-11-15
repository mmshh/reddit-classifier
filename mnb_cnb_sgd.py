import numpy as np
from collections import defaultdict
import nltk
import random
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import operator
from sklearn.model_selection import train_test_split
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB, BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline

from utils import convert_to_csv, read_train_data, read_test_data
nltk.download('stopwords')


def unique_comments(comment, result):
    classes_name, classes_count = np.unique(comment, return_index=True)
    tag = []
    
    for i in classes_count:
        tag.append(result[i])
    
    return classes_name, tag


def classes_to_integer(tag):
    lab = []
    classes_name = np.unique(tag)
    for i in range(len(tag)):
        lab.append(np.where(classes_name == tag[i])[0][0])
    lab = np.asarray(lab)
    
    return classes_name, lab


def get_datas():
    train_data = read_train_data()
    test_data = read_test_data()
    comment = train_data[0]
    result = train_data[1]
    
    comment_unique, result_unique = unique_comments(comment, result)
    classes_name, result_unique_integer = classes_to_integer(result_unique)
    
    return comment_unique, result_unique_integer, test_data, classes_name


def get_data_for_testing():   
    comment_unique, result_unique_integer, test_data, classes_name = get_datas()
    
    X_train, X_test, y_train, y_test = train_test_split(
     comment_unique, result_unique_integer, test_size=0.15, random_state=None)
    
    return X_train, X_test, y_train, y_test, classes_name


def train_predict_multNB(X_train, y_train, X_test, a):
    text_clf = Pipeline([
         ('vect', CountVectorizer()),
         ('tfidf', TfidfTransformer()),
         ('clf', MultinomialNB(alpha=a)),
    ])
    
    ovr = OneVsRestClassifier(text_clf)
    ovr.fit(X_train, y_train)  
    
    pred1 = ovr.predict_proba(X_test)
    return pred1    


def train_predict_compNB(X_train, y_train, X_test, a):
    text_clf2 = Pipeline([
         ('vect', CountVectorizer()),
         ('tfidf', TfidfTransformer()),
         ('clf', ComplementNB(alpha=a)),
    ])
    
    ovr2 = OneVsRestClassifier(text_clf2)
    ovr2.fit(X_train, y_train)  
    
    pred2 = ovr2.predict_proba(X_test)
    return pred2


def train_predict_sgd(x_t, y_t, x_test, a):
    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='log', penalty='elasticnet',alpha=a, random_state=42,
                            fit_intercept = False,  tol=0.00001, early_stopping=True, 
                                         n_iter_no_change=400, max_iter=20)),
                   ])
    sgd.fit(x_t, y_t)
    
    y_pred = sgd.predict_proba(x_test)
    return y_pred


def score_c(predictions, result):
    count = 0
    for i in range(len(predictions)):
        if(predictions[i] == result[i]):
            count += 1
    return 100*count/len(predictions)


def get_score_testing():
    X_train, X_test, y_train, y_test, classes_name = get_data_for_testing()
    pred1 = train_predict_multNB(X_train, y_train, X_test, 0.15)
    pred2 = train_predict_compNB(X_train, y_train, X_test, 0.24)
    pred3 = train_predict_sgd(X_train, y_train, X_test, 0.00001)
    
    pred = 15*pred1+10*pred2+6*pred3
    
    p  = [] 
    for i in range(len(pred)):
        p.append(classes_name[np.argmax(pred[i])])
        
    y  = [] 
    for i in range(len(y_test)):
        y.append(classes_name[y_test[i]])
    
    return y, p
    return score_c(y, p)



def write_predictions():
    X_train, y_train, X_test, classes_name = get_datas()
    pred1 = train_predict_multNB(X_train, y_train, X_test, 0.15)
    pred2 = train_predict_compNB(X_train, y_train, X_test, 0.24)
    pred3 = train_predict_sgd(X_train, y_train, X_test, 0.00001)
    
    pred = 15*pred1+10*pred2+6*pred4
    
    p  = [] 
    for i in range(len(pred)):
        p.append(classes_name[np.argmax(pred[i])])
        
    res = []
    for i in range(len(p)):
        res.append({'Id': i, 'Category': p[i]})
        
    convert_to_csv(res)   
    
if __name__ == "__main__":
    write_predictions()
    #print(get_score_testing()

