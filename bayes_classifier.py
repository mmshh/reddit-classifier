import numpy as np
from collections import defaultdict
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import operator

from utils import convert_to_csv, read_train_data, read_test_data


class BayesClassifier:
    def __init__(self):
        self.prior = defaultdict(float)
        self.word_frequency_class = defaultdict(lambda: defaultdict(int))
        self.word_probability_class = defaultdict(lambda: defaultdict(float))
        self.word_count_class = defaultdict(int)

    def train(self, comments, classes):
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
        
        #Compute word probability for a given class
        for c in self.word_frequency_class.keys():
            for word in self.word_frequency_class[c].keys():
                self.word_probability_class[c][word] = self.word_frequency_class[c][word] / self.word_count_class[c]
            
    def predict(self, comments):
        result = []
        for i in range(len(comments)):
            class_probability = defaultdict(float)
            words = self.tokenize_words(comments[i])
            for class_name in self.prior.keys():
                probX_knowingC = self.compute_probX_knowingC(class_name, words)
                class_probability[class_name] = probX_knowingC*self.prior[class_name]
            result.append({'Id': i, 'Category': max(class_probability.items(), key=operator.itemgetter(1))[0]})
        return result
    
    def tokenize_words(self, words):
        #Tokenize without ponctuation
        tokenizer = RegexpTokenizer(r'\w+')
        word_tokens = tokenizer.tokenize(words.lower())
        #Remove stop words
        stop_words = set(stopwords.words('english'))
        filtered_words = [w for w in word_tokens if not w in stop_words]
        return filtered_words
    
    def compute_probX_knowingC(self, class_name, words):
        #If no word exists in class return 0
        if (sum(self.word_probability_class[class_name][word] for word in words) == 0):
            return 0
        
        #Else return the product of P(word|c) for all word in words
        product_prob_x = 1.0
        for word in words:
            product_prob_x *= self.word_probability_class[class_name][word]
            #if(self.word_probability_class[class_name][word] != 0):
                #product_prob_x *= self.word_probability_class[class_name][word]
        return product_prob_x


if __name__ == "__main__":
    train_data = read_train_data()
    test_data = read_test_data()
    bayes_classifier = BayesClassifier()
    bayes_classifier.train(train_data[0][:], train_data[1][:])
    predictions = bayes_classifier.predict(test_data)
    convert_to_csv(predictions)
