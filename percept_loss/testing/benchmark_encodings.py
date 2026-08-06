from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import numpy as np

from .encoded_dataset import make_encodings

# bump when the probe protocol changes in a way that makes numbers incomparable.
#   1  raw (unstandardised) features, single 67/33 eval split  -- everything in FINDINGS.md §1
#   2  StandardScaler + logistic-regression probe + three-way fit/select/report split
# recorded in each run's config.json so a table can tell it is mixing protocols.
PROBE_VERSION = 2

def random_GaussianNB_test(data_loader, autoencoder, device):
    # quickest to train and test for dev purposes
    X, y = make_encodings(data_loader, autoencoder, device)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {'NB': acc}

def test_all_classifiers(data=None, autoencoder=None, device=None, data_loader=None, verbose=False):
    '''
    fit cheap probes on frozen encodings.

    the encodings of the *test* split are cut three ways, always with the same seed, so probe
    sample size is identical in every cell of the grid -- the data_percent axis moves only what
    the autoencoder saw:

        fit     67%    train the probe
        select  16.5%  choose the epoch  -> reported as '{name} select'
        report  16.5%  the number to quote -> reported as '{name}'

    the fit split is byte-identical to the old 67/33 split (same call, same seed); only the old
    33% eval half is subdivided. so '{name}' differs from the historical column solely by being
    measured on half as many instances.

    **why three ways.** picking the best epoch off the reported column is selection on the test
    set: with ~16 evals and a noise floor of a few points it inflates every number, and inflates
    noisy runs most. selecting on 'select' and reporting on 'report' is unbiased early stopping
    -- the two halves are disjoint. cost is variance: ~3k instances gives a standard error near
    0.9 percentage points. use `--select early_stop` in summarise_runs/plot_data_efficiency.

    'Linear' is logistic regression: the linear probe is the protocol the self-supervised
    literature reports (SimCLR, BYOL, SimSiam), so it is the number that makes results here
    comparable to published ones. 'MLP' is a 100-unit hidden layer -- a *non*-linear probe, and
    not the same claim.
    '''
    random_state=42
    classifiers = {
        'KNN': KNeighborsClassifier(),
        # max_iter is generous: lbfgs on 384 correlated latent dims does hit the limit at 1000,
        # and a probe that stops early under-fits the harder latents preferentially
        'Linear': LogisticRegression(max_iter=2000, random_state=random_state),
        # 'SVM-linear': SVC(kernel="linear", C=0.025, random_state=random_state),
        # 'SVM': SVC(gamma=2, C=1, random_state=random_state),
        # 'GP': GaussianProcessClassifier(1.0 * RBF(1.0), random_state=random_state),
        # 'Tree': DecisionTreeClassifier(random_state=random_state),
        # 'RF': RandomForestClassifier(random_state=random_state),
        'MLP': MLPClassifier(alpha=1, max_iter=1000, random_state=random_state),
        # 'ADA-Boost': AdaBoostClassifier(random_state=random_state),
        'NB': GaussianNB(),
        # 'QDA': QuadraticDiscriminantAnalysis(),
    }
    if data == None:
        X, y = make_encodings(data_loader, autoencoder, device)
    else:
        X, y = data
    X_train, X_held, y_train, y_held = train_test_split(X, y, test_size=0.33, random_state=42)
    X_select, X_test, y_select, y_test = train_test_split(X_held, y_held, test_size=0.5,
                                                          random_state=42)

    # standardise, fit on the probe's train split only.
    # KNN is raw euclidean distance and MLPClassifier(alpha=1) is heavily L2-regularised, so both
    # are scale-sensitive. that was harmless while every encoder ended in the same Tanh; it is
    # not now -- the VAE's mu is deliberately unbounded while the conv nets are bounded to
    # [-1, 1], so without this an architecture comparison partly measures latent scale.
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_select, X_test = scaler.transform(X_select), scaler.transform(X_test)

    # if verbose == True:
    #     print(f'made encoded dataset')
    #     for name, labels in zip(['train', 'test'], [y_train, y_test]):
    #         unique, counts = np.unique(labels, return_counts=True)
    #         print(f'{name} label proportions \n{np.asarray((unique, counts)).T}')

    data = []
    for name, clf in classifiers.items():
        data.append((name, clf, X_train, X_test, X_select, y_train, y_test, y_select, verbose))

    # with multiprocessing.Pool() as pool:
    #     mult_results = pool.imap(run_single, data)

    mult_results = map(run_single, data)

    results = {}
    for res in mult_results:
        name, acc, acc_select = res
        results[name] = acc                      # report split -- the number to quote
        results[f'{name} select'] = acc_select    # epoch-selection split -- never quote this
    return results

def run_single(data):
    name, clf, X_train, X_test, X_select, y_train, y_test, y_select, verbose = data
    clf.fit(X_train, y_train)

    acc_select = accuracy_score(y_select, clf.predict(X_select))
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    if verbose == True:
        print(f'clf: {name} = {acc*100}')
    return (name, acc, acc_select)