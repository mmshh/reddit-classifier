
# coding: utf-8

# In[1]:
import os

import numpy as np
import csv


# In[42]:
from utils import convert_to_csv, read_test_data


def classify_randomly(data):
    subreddits = ['AskReddit', 'GlobalOffensive', 'Music', 'Overwatch', 'anime',
       'baseball', 'canada', 'conspiracy', 'europe', 'funny',
       'gameofthrones', 'hockey', 'leagueoflegends', 'movies', 'nba',
       'nfl', 'soccer', 'trees', 'worldnews', 'wow']
    np.random.seed(0)
    random_ints = np.random.randint(0, 20, len(data))
    print(np.bincount(random_ints))
    predictions = []
    for idx in range(len(data)):
        predictions.append({'Id': idx, 'Category': subreddits[random_ints[idx]]})

    return predictions


def classify_2(data):
    subreddits = ['AskReddit', 'GlobalOffensive', 'Music', 'Overwatch', 'anime',
       'baseball', 'canada', 'conspiracy', 'europe', 'funny',
       'gameofthrones', 'hockey', 'leagueoflegends', 'movies', 'nba',
       'nfl', 'soccer', 'trees', 'worldnews', 'wow']
    
    np.random.seed(0)
    random_ints = np.random.randint(0, 20, len(data))
    print(np.bincount(random_ints))
    predictions = []
    
    
    for i in range(len(data)):
        result = np.zeros(20)
        subText = data[i].split()
        for j in range(len(subText)):
            if (subText[j] in subreddits):
                result[subreddits.index(subText[j])] += 1
        if(np.any(result)):
            predictions.append({'Id': i, 'Category': subreddits[np.argmax(result)]})
        else:
            predictions.append({'Id': i, 'Category': subreddits[random_ints[i]]})

    return predictions


if __name__ == "__main__":
    X_test = read_test_data()
    predictions = classify_2(X_test)
    convert_to_csv(predictions)

