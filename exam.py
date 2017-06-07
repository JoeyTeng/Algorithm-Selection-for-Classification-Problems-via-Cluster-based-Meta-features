# @Author: Joey Teng <JoeyTengDev>
# @Date:   05-Jun-2017
# @Email:  joey.teng.dev@gmail.com
# @Filename: exam.py
# @Last modified by:   JoeyTengDev
# @Last modified time: 07-Jun-2017



from sys import argv
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import json

class Models(object):
    def __init__(self):
        pass

    @staticmethod
    def naive_bayes(X, y):
        model = GaussianNB()
        return ('Gaussian Bayes', model)

    @staticmethod
    def knn(X, y):
        def score(self, X, y, sample_weight=None):
            self.set_params(n_neighbors=len(X))
            return self._score__(X, y, sample_weight)
        model = KNeighborsClassifier(n_neighbors=len(X), weights='distance', n_jobs=-1)
        KNeighborsClassifier._score__, KNeighborsClassifier.score = KNeighborsClassifier.score, score

        return ('K-Nearest Neighbors (hacked, All neighbors)', model)

    @staticmethod
    def decision_tree(X, y):
        model = DecisionTreeClassifier()
        return ('Decision Tree', model)

    @staticmethod
    def random_forest(X, y):
        model = RandomForestClassifier()
        return ('Support Vector Machine', model)

def main(fname):
    dataset = np.loadtxt(fname=fname, dtype=float, delimiter=',')
    X = dataset[:, :-1]
    y = dataset[:, -1]

    X = preprocess(X)
    for func in Models.__dict__.keys():
        if (func[0] == '_'):
            continue
        description, model = getattr(Models, func)(X, y)
        report(model, X, y, description)

def preprocess(X):
    normalized_X = preprocessing.normalize(X)
    return normalized_X

def report(model, X, y, description):
    scores = model_selection.cross_val_score(model, X, y, cv=10, n_jobs=-1)
    print()
    print(description)
    print(sum(scores) / len(scores))

if __name__ == '__main__':
#    fname = json.load(argv[1])
#    for entry in fname:
#        main(entry[0])
    main(argv[1])
