# Reddit Classifier #

### Description ###

This application classifies comments by subbredit using a dataset belonging to a Kaggle competition.
There are 20 subreddits and running the classifier will produce a file containing the class predictions.

#### Kaggle Team Name ####

Libra

### How to run the Bayes Classifier ###

To run the classifier against the `data_test.pkl` file from Kaggle that can beat all three baselines (Random, Bayes and Bayes with smoothing), 
please run the python file called `bayes_classifier_with_smoothing.py` located in the root directory of this project:
```
python bayes_classifier_with_smoothing.py
```
After completion, the generated file will be called `output.csv` and will be located in the resources folder. 

Note: Python 3 is required to run this project

### Used Libraries ###

- Numpy
- NLTK

### Developers ###

Jessica Gauvin

Maziar Mohammad-Shahi

