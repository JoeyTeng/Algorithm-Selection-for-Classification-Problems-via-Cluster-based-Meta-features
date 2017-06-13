# @Author: Joey Teng <JoeyTengDev>
# @Date:   05-Jun-2017
# @Email:  joey.teng.dev@gmail.com
# @Filename: exam.py
# @Last modified by:   Toujour
# @Last modified time: 14-Jun-2017



from sys import argv
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
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

class MetaFeature(object):
    def __init__(self, X, y):
        self.feature = {}
        self._X = X
        self._y = y

    def __universal_setter__(self, key, func=None):
        try:
            return self.feature[key]
        except KeyError:
            self.feature[key] = func()
            return self.feature[key]

    @property
    def N(self):
        """
        Number of examples
        """
        return self.__universal_setter__('N', lambda:len(self._y))

    @property
    def p(self):
        """
        Number of attributes
        """
        return self.__universal_setter__('p', lambda:len(self._X[0]))

    @property
    def q(self):
        """
        Number of classes
        """
        return self.__universal_setter__('q', lambda:len(np.unique(self._y)))

    @property
    def Bin_att(self):
        """
        Number of binary attributes
        """
        return self.__universal_setter__('Bin.att', lambda:len([i for i in np.transpose(self._X) if len(np.unique(i)) <= 2]))

    @property
    def cost(self):
        """
        TODO:

        Cost of matrix indicator
        """
        pass

    @property
    def SD(self):
        """
        TODO:

        Standard deviation ration (geometric mean)
        """
        pass

    @property
    def corr_abs(self):
        """
        TODO:

        Mean absolute correlation of attributes
        """
        pass

    @property
    def cancor1(self):
        """
        TODO:

        First canonical correlation
        """
        pass

    @property
    def fract1(self):
        """
        TODO:

        Fraction separability due to cancor1
        """
        pass

    @property
    def skewness(self):
        r"""
        Mean of $\E (X - \mu)^3\/\sigma^3$
        """
        return self.__universal_setter__('skewness', lambda:stats.skew(self._X, bias=False)) # bias correction enabled. CHECK!

    @property
    def kurtosis(self):
        r"""
        Mean of $\E(X - \mu)^4\/\sigma^4$
        """
        return self.__universal_setter__('kurtosis', lambda:stats.kurtosis(self._X, fisher=True, bias=False)) # fisher = true (normal = 0), bias correction enabled. CHECK!

    @property
    def H_C(self):
        r"""
        TODO:

        $H(C)$
        Entropy (complexity) of class
        """
        pass

    @property
    def H_bar_X(self):
        r"""
        TODO:

        $\bar H(X)$
        Mean entropy (complexity) of attributes
        """
        pass

    @property
    def M_bar_C_X(self):
        r"""
        TODO:

        $\bar M(C, X)$
        Mean mutual information of class and attributes
        """
        pass

    @property
    def EN_attr(self):
        r"""
        TODO:

        Equivalent number of attributes $H(C)/\bar M(C, X)$
        """
        pass

    @property
    def NS_ratio(self):
        r"""
        TODO:

        Noise-signal ratio (\bar H  (X) − \bar M (C, X))/\bar M (C, X)
        """
        pass

def main(fname):
    dataset = np.loadtxt(fname=fname, dtype=float, delimiter=',')
    X = dataset[:, :-1]
    y = dataset[:, -1]

    X = preprocess(X)

    scores = {}
    for func in Models.__dict__.keys():
        if (func[0] == '_'):
            continue
        description, model = getattr(Models, func)(X, y)
        scores[func] = report(model, X, y, description)

    paired_t_test(scores)

    meta_feature(X, y)

def meta_feature(X, y):
    feature = MetaFeature(X, y)
    for func in feature.__dir__():
        if (func[0] != '_'):
            print(func)
            print(getattr(feature, func))

def paired_t_test(scores):
    print("paired t-test %r" %(scores.keys()))
    result = list(map((lambda x:[stats.ttest_rel(x, b) for b in scores.values()]), scores.values()))
    p_value = np.array(([x[1] for x in row] for row in result))
    print(p_value)
    # print(result)
    return result

def preprocess(X):
    normalized_X = preprocessing.normalize(X)
    return normalized_X

def report(model, X, y, description):
    scores = model_selection.cross_val_score(model, X, y, cv=10, n_jobs=-1)
    print()
    print(description)
    print(sum(scores) / len(scores))

    return scores

if __name__ == '__main__':
#    fname = json.load(argv[1])
#    for entry in fname:
#        main(entry[0])
    main(argv[1])
