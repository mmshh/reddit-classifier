import numpy as np
import random
from collections import defaultdict
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import operator
class BayesClassifier:
    def __init__(self):
        self.prior = defaultdict(float)
        self.voc = []
        self.word_frequency_class = defaultdict(lambda: defaultdict(int))
        self.word_count_class = defaultdict(int)

    def train(self, comments, classes):
        print('start training')
        classes_name, classes_count = np.unique(classes, return_counts=True)
        
        #Compute prior
        for i in range(len(classes_name)):
            self.prior[classes_name[i]] = classes_count[i]/len(classes)
        
        #Compute frequencies and class counts
        for i in range(len(comments)):
            bag = self.tokenize_words(comments[i])
            for w in bag:
                self.word_frequency_class[classes[i]][w] += 1
                self.word_count_class[classes[i]] += 1
                if(w not in self.voc):
                    self.voc.append(w)
            
    def predict(self, comments, alpha):
        print('start predict')
        result = []
        for i in range(len(comments)):
            class_probability = defaultdict(float)
            words = self.tokenize_words(comments[i])
            for class_name in self.prior.keys():
                probX_knowingC = self.compute_probX_knowingC(class_name, words, alpha)
                class_probability[class_name] = probX_knowingC + np.log(self.prior[class_name])
            result.append({'Id': i, 'Category': max(class_probability.items(), key=operator.itemgetter(1))[0]})
        return result
    
    def tokenize_words(self, words):
        #Tokenize without ponctuation
        tokenizer = RegexpTokenizer(r'\w+')
        word_tokens = tokenizer.tokenize(words.lower())
        #Remove stop words
        stop_words = set(stopwords.words('english'))
        ps = nltk.PorterStemmer()
        filtered_words = [ps.stem(w) for w in word_tokens
                          if w not in stop_words and not w.isdigit()]
        return filtered_words
    
    def compute_probX_knowingC(self, class_name, words, alpha):      
        #Return the product of P(word|c) for all word in words
        product_prob_x = 0.
        for word in words:
            product_prob_x += np.log(self.compute_probx_knowingC(class_name, word, alpha))
        return product_prob_x
    
    def compute_probx_knowingC(self, class_name, word, alpha):
        #Compute word probability for a given class
        return (self.word_frequency_class[class_name][word]+alpha) / (self.word_count_class[class_name]+alpha*len(self.voc))

    
    def split_data(self, data_c, data_r, train_percent):
        data_comments = np.array(data_c)
        data_class = np.array(data_r)
        random_indices = np.array(random.sample(range(len(data_c)), int(len(data_c)*train_percent)))
        return data_comments[random_indices], data_class[random_indices], data_comments[~random_indices], data_class[~random_indices]
        
        
    def score(self, predictions, result):
        count = 0
        for i in range(len(predictions)):
            if(predictions[i]['Category'] == result[i]):
                count += 1
        return count/len(predictions)

    def define_alpha(self, validation_comments, validation_result):
        alpha = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
        result = np.zeros(len(alpha))
        for i in range(len(alpha)):
            print('Alpha ',i+1,'/',len(alpha),' : ',alpha[i])
            predict = bayes_classifier.predict(v_c, alpha[i])
            result[i] = bayes_classifier.score(predict, v_r)
            print(result[i])
            if(i>0 and result[i]<result[i-1]):
                break
        print(result)
        print(alpha[np.argmax(result)])
        return alpha[np.argmax(result)]
        

if __name__ == "__main__":
    train_data = np.load("./resources/data_train.pkl", allow_pickle=True)
    test_data = np.load("./resources/data_test.pkl", allow_pickle=True)
    comment = train_data[0]
    result = train_data[1]
    bayes_classifier = BayesClassifier()
    train_comment, train_result, test_comment, test_result = bayes_classifier.split_data(comment, result, 0.8)
    #train_comment = comment[:50000] 
    #train_result = result[:50000]
    #test_comment = comment[50000:]
    #test_result = result[50000:]
    #t_c, t_r, v_c, v_r = bayes_classifier.split_data(train_comment, train_result, 0.7)
    #bayes_classifier.train(t_c, t_r)
    #alpha_star = bayes_classifier.define_alpha(v_c, v_r)
    alpha_star = 0.01
    bayes_classifier.train(train_comment, train_result)
    predictions = bayes_classifier.predict(test_comment, alpha_star)
    print(bayes_classifier.score(predictions, test_result))
