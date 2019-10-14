import numpy as np
from collections import defaultdict
import nltk
import operator
import csv



class BayesClassifier:
    def __init__(self):
        self.prior = defaultdict(float)
        self.word_frequency_class = defaultdict(lambda: defaultdict(int))
        self.word_probability_class = defaultdict(lambda: defaultdict(float))
        self.word_count_class = defaultdict(int)

    def train(self, comments, classes):
        classes_name, classes_count = np.unique(classes, return_counts=True)
        
        for i in range(len(classes_name)):
            self.prior[classes_name[i]] = classes_count[i]/len(classes)
        
        for i in range(len(comments)):
            bag = nltk.tokenize.word_tokenize(comments[i])
            for w in bag:
                self.word_frequency_class[classes[i]][w] += 1
                self.word_count_class[classes[i]] += 1

        for c in self.word_frequency_class.keys():
            for word in self.word_frequency_class[c].keys():
                self.word_probability_class[c][word] = self.word_frequency_class[c][word] / self.word_count_class[c]
            
    def predict(self, comments):
        result = []
        for i in range(len(comments)):
            class_probability = defaultdict(float)
            words = nltk.tokenize.word_tokenize(comments[i])
            for class_name in self.prior.keys():
                product = 1.0
                for word in words:
                    product *= self.word_probability_class[class_name][word]
                class_probability[class_name] = self.prior[class_name]*product
            result.append({'Id': i, 'Category': max(class_probability.items(), key=operator.itemgetter(1))[0]})
        return result
            
    def convert_to_csv(self, data):
        csv_columns = ['Id', 'Category']
        with open('./resources/output2.csv', 'w') as f:
            writer = csv.DictWriter(f, csv_columns)
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    train_data = np.load("./resources/data_train.pkl", allow_pickle=True)
    test_data = np.load("./resources/data_test.pkl", allow_pickle=True)
    bayes_classifier = BayesClassifier()
    bayes_classifier.train(train_data[0][:], train_data[1][:])
    predictions = bayes_classifier.predict(test_data[0][:])
    bayes_classifier.convert_to_csv(predictions)
