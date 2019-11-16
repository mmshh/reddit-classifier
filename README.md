# Reddit Classifier #

### Description ###

This application classifies comments by subbredit using a dataset belonging to a Kaggle competition.
There are 20 subreddits and running the classifier will produce a file containing the class predictions.

#### Kaggle Team Name ####

Libra

### How to run the Bayes Classifier ###

The code for the models in the report is all included. To run the best classifier against the `data_test.pkl` file from Kaggle that produces the best accuracy, 
please run the python file called `mnb_cnb_sgd.py` located in the root directory of this project:
```
python mnb_cnb_sgd.py
```
After completion, the generated file will be called `output.csv` and will be located in the resources folder. 

Note: Python 3 is required to run this project

### Required Libraries to run  ###

- Scikit-learn
- Numpy
- NLTK

### Developers ###

Jessica Gauvin

Maziar Mohammad-Shahi

