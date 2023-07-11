from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import multiprocessing

from .encoded_dataset import make_encodings

def random_forest_test(data_loader, autoencoder, device):
    X, y = make_encodings(data_loader, autoencoder, device)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

def test_all_classifiers(data_loader, autoencoder, device, verbose=False):
    random_state=42
    classifiers = {
        'KNN': KNeighborsClassifier(3),
        # 'SVM-linear': SVC(kernel="linear", C=0.025, random_state=random_state),
        # 'SVM': SVC(gamma=2, C=1, random_state=random_state),
        # 'GP': GaussianProcessClassifier(1.0 * RBF(1.0), random_state=random_state),
        'Tree': DecisionTreeClassifier(random_state=random_state),
        'RF': RandomForestClassifier(random_state=random_state),
        'MLP': MLPClassifier(alpha=1, max_iter=1000, random_state=random_state),
        'ADA-Boost': AdaBoostClassifier(random_state=random_state),
        'NB': GaussianNB(),
        # 'QDA': QuadraticDiscriminantAnalysis(),
    }

    X, y = make_encodings(data_loader, autoencoder, device)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    if verbose == True:
        print(f'made encoded dataset')
        for name, labels in zip(['train', 'test'], [y_train, y_test]):
            unique, counts = np.unique(labels, return_counts=True)
            print(f'{name} label proportions \n{np.asarray((unique, counts)).T}')

    data = []
    for name, clf in classifiers.items():
        data.append((name, clf, X_train, X_test, y_train, y_test, verbose))

    # with multiprocessing.Pool() as pool:
    #     mult_results = pool.imap(run_single, data)

    mult_results = map(run_single, data)

    results = {}
    for res in mult_results:
        name, acc = res
        results[name] = acc
    return results

def run_single(data):
    name, clf, X_train, X_test, y_train, y_test, verbose = data
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    if verbose == True:
        print(f'clf: {name} = {acc*100}')
    return (name, acc)