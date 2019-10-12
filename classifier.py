import numpy as np
import csv


def read_test_data():
    return np.load("./resources/data_test.pkl", allow_pickle=True)


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


def convert_to_csv(data):
    csv_columns = ['Id', 'Category']
    with open('./resources/output.csv', 'w') as f:
        writer = csv.DictWriter(f, csv_columns)
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    X_test = read_test_data()
    predictions = classify_randomly(X_test)
    convert_to_csv(predictions)