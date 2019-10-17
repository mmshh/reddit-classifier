import numpy as np
from collections import defaultdict
import nltk
import random
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import operator

from utils import convert_to_csv, read_train_data, read_test_data

nltk.download('stopwords')


class BayesClassifier:
    def __init__(self):
        self.prior = defaultdict(float)
        self.voc = []
        self.word_frequency_class = defaultdict(lambda: defaultdict(int))
        self.word_count_class = defaultdict(int)

    def train(self, comments, classes):
        """
        Computes the prior and the word frequency in each class and total word count in each class
        :param comments: raw comments
        :param classes: classes
        :return: void
        """
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
        """
        Predicts the classes for the comments passed as the output using the alpha as the hyper parameter
        :param comments: raw comments
        :param alpha: the smoothing hyper param
        :return: class predictions
        """
        print('start predictions')
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
        """
        Tokenizes, removes the stop words and digits and outputs the stems of the words to produce the bag of words
        :param words: the comments
        :return: the bag of words
        """
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
        """
        Returns the log probability of P(word|class) for all words in the bag of words
        :param class_name: class name
        :param words: bag of words
        :param alpha: hyper param
        :return: P(word|c)
        """
        product_prob_x = 0.
        for word in words:
            product_prob_x += np.log(self.compute_probx_knowingC(class_name, word, alpha))
        return product_prob_x

    def compute_probx_knowingC(self, class_name, word, alpha):
        #Compute word probability for a given class
        return (self.word_frequency_class[class_name][word]+alpha) / (self.word_count_class[class_name]+alpha*len(self.voc))

    def split_data(self, data_c, data_r, train_percent):
        """
        Helper function to validate the data for validation
        """
        data_comments = np.array(data_c)
        data_class = np.array(data_r)
        random_indices = np.array(random.sample(range(len(data_c)), int(len(data_c)*train_percent)))
        return data_comments[random_indices], data_class[random_indices], data_comments[~random_indices], data_class[~random_indices]

    def score(self, predictions, result):
        """
        Helper function to check accuracy for validation
        """
        count = 0
        for i in range(len(predictions)):
            if(predictions[i]['Category'] == result[i]):
                count += 1
        return count/len(predictions)

    def define_alpha(self, validation_comments, validation_result):
        """
        Helper function to find a good value for hyper param alpha
        """
        alpha = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
        result = np.zeros(len(alpha))
        for i in range(len(alpha)):
            print('Alpha ',i+1,'/',len(alpha),' : ',alpha[i])
            predict = bayes_classifier.predict(validation_comments, alpha[i])
            result[i] = bayes_classifier.score(predict, validation_result)
            print(result[i])
        print(result)
        print(alpha[np.argmax(result)])
        return alpha[np.argmax(result)]


if __name__ == "__main__":
    train_data = read_train_data()
    test_data = read_test_data()
    comment = train_data[0]
    result = train_data[1]
    bayes_classifier = BayesClassifier()
    alpha_star = 0.01
    bayes_classifier.train(comment, result)
    predictions = bayes_classifier.predict(test_data, alpha_star)
    convert_to_csv(predictions)
