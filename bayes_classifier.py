import numpy as np
from collections import defaultdict
import nltk


class BayesClassifier:
    def __init__(self, prior):
        self.prior = prior
        self.word_frequency = defaultdict(int)
        self.word_probability = defaultdict(float)
        self.word_count = 0

    def train(self, comments):
        for comment in comments:
            bag = nltk.tokenize.word_tokenize(comment)
            for w in bag:
                self.word_frequency[w] += 1
                self.word_count += 1

        for word in self.word_frequency.keys():
            self.word_probability[word] = self.word_frequency[word] / self.word_count


if __name__ == "__main__":
    data = np.load("./resources/data_train.pkl", allow_pickle=True)
